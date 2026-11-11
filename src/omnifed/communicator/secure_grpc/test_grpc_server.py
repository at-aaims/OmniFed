#!/usr/bin/env python3
"""Secure (mutual-TLS) gRPC aggregation server -- standalone test.

Clients send random float32 arrays; once every allowed client has submitted for a
round, the server returns the sample-weighted mean  sum(n_i * x_i) / sum(n_i)  to
each of them.  Nothing here imports OmniFed, protobuf or torch: only grpcio + numpy.

What is different from the default OmniFed gRPC communicator (see SECURE_GRPC.md):

  * Transport: mTLS only (add_secure_port, require_client_auth=True).  There is no
    insecure port.
  * Identity: taken from the TLS client certificate via context.auth_context()
    (the certificate CN) and checked against --allowed-clients.  The request carries
    NO client_id field, so a caller cannot claim to be someone else.
  * One contribution per identity per round; re-sends are idempotent (no double
    counting) and a lost response can be replayed for the previous round.
  * The SERVER owns the round counter; a client cannot advance or reset it.
  * Strict validation: tensor names / shapes / dtype must equal --shapes, values must
    be finite, num_samples bounded, sizes checked BEFORE any allocation.
  * Deadlines: every handler waits at most min(--round-timeout, client deadline);
    clients that set no deadline are rejected unless --allow-missing-deadline.
  * Concurrency: --max-concurrent-rpcs (hard cap, excess gets RESOURCE_EXHAUSTED),
    --max-workers, per-connection --max-concurrent-streams, --max-message-mb,
    idle-connection reaping and keepalive pings.
  * Wire format: JSON header + raw float32 bytes (no pickle, no torch.load).

Run:  python test_grpc_server.py --ca certs/ca.crt --cert certs/server/server.crt \
          --key certs/server/server.key --allowed-clients client0,client1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import ssl
import stat
import struct
import sys
import threading
import time
from concurrent import futures
from math import prod

import grpc
import numpy as np

log = logging.getLogger("secure-grpc-server")

# --------------------------------------------------------------------------------------
# Wire format (keep identical in test_grpc_client.py)
#   frame  = MAGIC | u32le header_len | header_json | raw little-endian float32 tensors
#   header = {"v": 1, "meta": {...}, "tensors": [{"name", "dtype", "shape"}, ...]}
# --------------------------------------------------------------------------------------
SERVICE = "omnifed.secure.Aggregator"
WIRE_MAGIC = b"OFSG"
WIRE_VERSION = 1
DTYPE = "float32"
ITEMSIZE = 4
MAX_HEADER_BYTES = 1 << 20
MAX_TENSORS = 4096
MAX_NDIM = 8
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-/]{1,256}$")
DEFAULT_SHAPES = "layer1.weight:256x128,layer1.bias:128,layer2.weight:64x256"


class WireError(ValueError):
    """Malformed or unacceptable frame."""


class Reject(Exception):
    """Domain error carrying the gRPC status *name* to return to the caller."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def parse_shapes(text: str) -> dict[str, tuple[int, ...]]:
    """'a:2x3,b:4' -> {'a': (2, 3), 'b': (4,)}"""
    spec: dict[str, tuple[int, ...]] = {}
    for item in text.split(","):
        name, _, dims = item.strip().partition(":")
        if not NAME_RE.match(name) or not dims:
            raise ValueError(f"bad --shapes entry {item!r}; expected name:AxB")
        shape = tuple(int(d) for d in dims.lower().split("x"))
        if not shape or len(shape) > MAX_NDIM or any(d < 1 for d in shape):
            raise ValueError(f"bad shape for {name!r}: {dims!r}")
        if name in spec:
            raise ValueError(f"duplicate tensor name {name!r}")
        spec[name] = shape
    return spec


def spec_nbytes(spec: dict[str, tuple[int, ...]]) -> int:
    return sum(prod(s) * ITEMSIZE for s in spec.values())


def encode_frame(meta: dict, tensors: dict) -> bytes:
    entries, chunks = [], []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype="<f4")
        entries.append({"name": name, "dtype": DTYPE, "shape": list(a.shape)})
        chunks.append(a.tobytes())
    header = json.dumps(
        {"v": WIRE_VERSION, "meta": meta, "tensors": entries}, separators=(",", ":")
    ).encode()
    return WIRE_MAGIC + struct.pack("<I", len(header)) + header + b"".join(chunks)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def decode_frame(buf: bytes, spec: dict[str, tuple[int, ...]]) -> tuple[dict, dict]:
    """Parse and strictly validate a frame against ``spec``.  Raises WireError.

    Names, dtype and shapes are compared with the expected spec before any array is
    built, so a peer cannot make us allocate by claiming a huge shape.
    """
    if len(buf) < 8 or buf[:4] != WIRE_MAGIC:
        raise WireError("bad magic")
    (hlen,) = struct.unpack_from("<I", buf, 4)
    if hlen > MAX_HEADER_BYTES or 8 + hlen > len(buf):
        raise WireError("bad header length")
    try:
        header = json.loads(bytes(buf[8 : 8 + hlen]))
    except (ValueError, UnicodeDecodeError):
        raise WireError("header is not valid JSON") from None
    if not isinstance(header, dict) or header.get("v") != WIRE_VERSION:
        raise WireError(f"unsupported wire version (want {WIRE_VERSION})")
    meta, entries = header.get("meta"), header.get("tensors")
    if not isinstance(meta, dict) or not isinstance(entries, list):
        raise WireError("malformed header")
    if len(entries) > MAX_TENSORS or len(entries) != len(spec):
        raise WireError("tensor count mismatch")

    offset = 8 + hlen
    out: dict[str, np.ndarray] = {}
    for e in entries:
        if not isinstance(e, dict):
            raise WireError("malformed tensor entry")
        name, dtype, shape = e.get("name"), e.get("dtype"), e.get("shape")
        if not isinstance(name, str) or name not in spec or name in out:
            raise WireError(f"unexpected or duplicate tensor {name!r}"[:200])
        if dtype != DTYPE:
            raise WireError(f"tensor {name!r}: dtype must be {DTYPE}")
        if (
            not isinstance(shape, list)
            or not all(_is_int(d) for d in shape)
            or tuple(shape) != spec[name]
        ):
            raise WireError(f"tensor {name!r}: shape must be {list(spec[name])}")
        n = prod(spec[name])
        if offset + n * ITEMSIZE > len(buf):
            raise WireError("truncated payload")
        arr = np.frombuffer(buf, dtype="<f4", count=n, offset=offset).reshape(spec[name])
        if not np.isfinite(arr).all():
            raise WireError(f"tensor {name!r} contains NaN/Inf")
        out[name] = arr
        offset += n * ITEMSIZE
    if offset != len(buf):
        raise WireError("trailing bytes")
    return meta, out


def validate_request_meta(meta: dict, max_samples: int) -> tuple[int, int]:
    rnd, nsamp = meta.get("round"), meta.get("num_samples")
    if not (_is_int(rnd) and 0 <= rnd < 2**31):
        raise WireError("meta.round must be an int in [0, 2^31)")
    if not (_is_int(nsamp) and 1 <= nsamp <= max_samples):
        raise WireError(f"meta.num_samples must be an int in [1, {max_samples}]")
    return rnd, nsamp


# --------------------------------------------------------------------------------------
# Numerics (float64 accumulation, float32 result)
# --------------------------------------------------------------------------------------
def _acc_new(spec):
    return {k: np.zeros(shape, dtype=np.float64) for k, shape in spec.items()}


def _acc_add(acc, tensors, weight: float) -> None:
    for k, arr in tensors.items():
        acc[k] += arr.astype(np.float64) * weight


def _acc_mean(acc, total: float):
    return {k: (v / total).astype(np.float32) for k, v in acc.items()}


# --------------------------------------------------------------------------------------
# Round state machine (no grpc inside: raises Reject, takes an is_active callable)
# --------------------------------------------------------------------------------------
class Aggregator:
    def __init__(self, spec, expected: int, rounds: int, round_timeout: float):
        self.spec = spec
        self.expected = expected
        self.rounds = rounds
        self.round_timeout = round_timeout
        self.cond = threading.Condition()
        self.round = 0  # round currently being collected (server-owned)
        self.generation = 0  # bumped when a partial round is abandoned
        self.acc = None
        self.submitted: dict[str, int] = {}  # identity -> num_samples
        self.started_at: float | None = None
        self.results: dict[int, tuple[bytes, frozenset]] = {}
        self.done = threading.Event()

    def wake(self) -> None:
        """Registered as an RPC-termination callback so cancelled waiters free their thread."""
        with self.cond:
            self.cond.notify_all()

    def _expire_stale_locked(self) -> None:
        if self.started_at is not None and time.monotonic() - self.started_at > self.round_timeout:
            log.warning(
                "round %d abandoned: %d/%d clients (%s) after %.0fs; dropping partial sums",
                self.round, len(self.submitted), self.expected,
                ",".join(sorted(self.submitted)), self.round_timeout,
            )
            self.acc, self.submitted, self.started_at = None, {}, None
            self.generation += 1
            self.cond.notify_all()

    def _finalize_locked(self) -> None:
        total = sum(self.submitted.values())
        blob = encode_frame(
            {"round": self.round, "participants": len(self.submitted), "total_samples": total},
            _acc_mean(self.acc, float(total)),
        )
        took = time.monotonic() - (self.started_at or time.monotonic())
        log.info(
            "round %d complete: %s total_samples=%d window=%.2fs result_sha256=%s",
            self.round, sorted(self.submitted), total, took, hashlib.sha256(blob).hexdigest()[:12],
        )
        self.results[self.round] = (blob, frozenset(self.submitted))
        for old in [k for k in self.results if k < self.round - 3]:
            del self.results[old]
        self.acc, self.submitted, self.started_at = None, {}, None
        self.round += 1
        if self.round >= self.rounds:
            self.done.set()
        self.cond.notify_all()

    def submit_and_wait(self, ident: str, rnd: int, nsamp: int, tensors, budget: float, is_active) -> bytes:
        with self.cond:
            self._expire_stale_locked()
            replay = self.results.get(rnd)
            if replay is not None and ident in replay[1]:
                log.info("round %d: replaying stored result to %s", rnd, ident)
                return replay[0]
            if self.round >= self.rounds:
                raise Reject("FAILED_PRECONDITION", "all rounds are finished")
            if rnd != self.round:
                raise Reject("FAILED_PRECONDITION", f"expected round {self.round}, got {rnd}")
            if ident not in self.submitted:
                if self.acc is None:
                    self.acc = _acc_new(self.spec)
                _acc_add(self.acc, tensors, float(nsamp))
                self.submitted[ident] = nsamp
                if self.started_at is None:
                    self.started_at = time.monotonic()
                log.info("round %d: %s submitted n=%d (%d/%d)", rnd, ident, nsamp,
                         len(self.submitted), self.expected)
                if len(self.submitted) == self.expected:
                    self._finalize_locked()
            else:
                log.info("round %d: repeat submit from %s ignored (idempotent)", rnd, ident)

            gen = self.generation
            self.cond.wait_for(
                lambda: self.round > rnd or self.generation != gen or not is_active(),
                timeout=budget,
            )
            if self.round > rnd:
                return self.results[rnd][0]
            if not is_active():
                return b""  # caller went away; response is discarded
            if self.generation != gen:
                raise Reject("ABORTED", "round was abandoned (straggler timeout); resend")
            raise Reject(
                "DEADLINE_EXCEEDED",
                f"round {rnd} incomplete ({len(self.submitted)}/{self.expected}) within {budget:.1f}s",
            )


# --------------------------------------------------------------------------------------
# Authentication / deadline helpers
# --------------------------------------------------------------------------------------
def _cert_fingerprint(ac) -> str | None:
    pems = ac.get("x509_pem_cert") or []
    try:
        return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pems[0].decode())).hexdigest()
    except (IndexError, ValueError, UnicodeDecodeError):
        return None


def authenticate(context, allowed: frozenset) -> str:
    """Caller identity = CN of the verified TLS client certificate (auth_context)."""
    peer = context.peer()
    ac = context.auth_context() or {}
    if list(ac.get("transport_security_type") or []) != [b"ssl"]:
        log.warning("DENY no TLS identity peer=%s", peer)
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "a TLS client certificate is required")
    cns = list(ac.get("x509_common_name") or [])
    if len(cns) != 1:
        log.warning("DENY certificate without exactly one CN peer=%s", peer)
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "client certificate must carry exactly one CN")
    ident = cns[0].decode("utf-8", "replace")
    if ident not in allowed:
        log.warning("DENY identity=%r not in allowlist peer=%s", ident, peer)
        context.abort(grpc.StatusCode.PERMISSION_DENIED, "identity not allowed")
    return ident


# grpcio does NOT return None from time_remaining() when the client set no deadline: it
# returns the C-core "infinite future" converted to seconds (~9.2e18; verified on 1.80 and
# the 1.81.1 source: max(call_details.deadline - time.time(), 0)).  Anything this large
# means "no deadline".
NO_DEADLINE_S = 1e12


def wait_budget(context, round_timeout: float, require_deadline: bool) -> float:
    """Seconds this handler may block: min(--round-timeout, client's remaining deadline)."""
    remaining = context.time_remaining()
    if remaining is None or remaining >= NO_DEADLINE_S:
        if require_deadline:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "RPC deadline required: set timeout= on the call")
        return round_timeout
    return min(round_timeout, max(0.0, remaining))


class AggregatorService:
    def __init__(self, agg: Aggregator, allowed: frozenset, max_samples: int, require_deadline: bool):
        self.agg, self.allowed = agg, allowed
        self.max_samples, self.require_deadline = max_samples, require_deadline

    def _authenticated(self, fn):
        """Every handler goes through here, so none can forget to authenticate."""
        def handler(request, context):
            ident = authenticate(context, self.allowed)
            return fn(ident, request, context)
        return handler

    def handlers(self) -> grpc.GenericRpcHandler:
        return grpc.method_handlers_generic_handler(SERVICE, {
            "Aggregate": grpc.unary_unary_rpc_method_handler(self._authenticated(self.aggregate)),
            "WhoAmI": grpc.unary_unary_rpc_method_handler(self._authenticated(self.who_am_i)),
        })

    def who_am_i(self, ident: str, request: bytes, context) -> bytes:
        ac = context.auth_context() or {}
        return json.dumps({"identity": ident, "peer": context.peer(),
                           "cert_sha256": _cert_fingerprint(ac)}).encode()

    def aggregate(self, ident: str, request: bytes, context) -> bytes:
        budget = wait_budget(context, self.agg.round_timeout, self.require_deadline)
        try:
            meta, tensors = decode_frame(request, self.agg.spec)
            rnd, nsamp = validate_request_meta(meta, self.max_samples)
        except WireError as e:
            log.warning("REJECT %s: %s", ident, e)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad request: {e}")
        context.add_callback(self.agg.wake)
        try:
            return self.agg.submit_and_wait(ident, rnd, nsamp, tensors, budget, context.is_active)
        except Reject as r:
            log.info("round %s for %s -> %s: %s", rnd, ident, r.code, r.detail)
            context.abort(getattr(grpc.StatusCode, r.code), r.detail)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _warn_if_key_open(path: str) -> None:
    try:
        if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            log.warning("private key %s is group/world accessible; chmod 600 it", path)
    except OSError:
        pass


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mTLS gRPC averaging server (standalone test)")
    p.add_argument("--bind", default="0.0.0.0:50051", help="host:port to listen on [0.0.0.0:50051]")
    p.add_argument("--ca", required=True, help="CA certificate used to verify client certificates")
    p.add_argument("--cert", required=True, help="server certificate (PEM)")
    p.add_argument("--key", required=True, help="server private key (PEM)")
    p.add_argument("--allowed-clients", required=True,
                   help="comma-separated certificate CNs that may participate")
    p.add_argument("--num-clients", type=int, default=0,
                   help="clients per round (default: number of allowed clients)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="name:AxB,... (must match clients)")
    p.add_argument("--round-timeout", type=float, default=300.0,
                   help="max seconds a handler waits for stragglers / a partial round lives [300]")
    p.add_argument("--allow-missing-deadline", action="store_true",
                   help="accept RPCs without a client deadline (default: reject them)")
    p.add_argument("--max-concurrent-rpcs", type=int, default=0,
                   help="hard cap on in-flight RPCs, extra ones get RESOURCE_EXHAUSTED [clients+4]")
    p.add_argument("--max-workers", type=int, default=0, help="thread-pool size [= max-concurrent-rpcs]")
    p.add_argument("--max-concurrent-streams", type=int, default=8, help="HTTP/2 streams per connection [8]")
    p.add_argument("--max-message-mb", type=int, default=64, help="send/receive message cap in MiB [64]")
    p.add_argument("--max-samples", type=int, default=10**9, help="largest accepted num_samples [1e9]")
    p.add_argument("--idle-timeout-sec", type=int, default=600, help="close idle connections after N s [600]")
    p.add_argument("--keepalive-sec", type=int, default=30, help="server keepalive ping interval [30]")
    p.add_argument("--shutdown-grace", type=float, default=10.0, help="seconds to let in-flight RPCs finish")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        spec = parse_shapes(args.shapes)
    except ValueError as e:
        log.error("%s", e)
        return 2
    allowed = frozenset(n.strip() for n in args.allowed_clients.split(",") if n.strip())
    if not allowed:
        log.error("--allowed-clients is empty")
        return 2
    expected = args.num_clients or len(allowed)
    if not (1 <= expected <= len(allowed)):
        log.error("--num-clients must be in [1, %d]", len(allowed))
        return 2
    cap = args.max_concurrent_rpcs or expected + 4
    if cap < expected:
        log.error("--max-concurrent-rpcs=%d < clients per round (%d): a round could never complete",
                  cap, expected)
        return 2
    workers = max(args.max_workers or cap, cap)  # a worker for every admitted RPC, else waiters can starve
    max_msg = args.max_message_mb * 1024 * 1024
    if spec_nbytes(spec) + MAX_HEADER_BYTES > max_msg:
        log.error("payload %.1f MiB exceeds --max-message-mb=%d; raise it", spec_nbytes(spec) / 2**20,
                  args.max_message_mb)
        return 2

    _warn_if_key_open(args.key)
    creds = grpc.ssl_server_credentials(
        [(_read(args.key), _read(args.cert))],
        root_certificates=_read(args.ca),
        require_client_auth=True,  # <- this is the "m" in mTLS
    )

    agg = Aggregator(spec, expected, args.rounds, args.round_timeout)
    service = AggregatorService(agg, allowed, args.max_samples, not args.allow_missing_deadline)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rpc"),
        maximum_concurrent_rpcs=cap,
        options=[
            ("grpc.max_receive_message_length", max_msg),
            ("grpc.max_send_message_length", max_msg),
            ("grpc.max_concurrent_streams", args.max_concurrent_streams),
            ("grpc.max_connection_idle_ms", args.idle_timeout_sec * 1000),
            ("grpc.keepalive_time_ms", args.keepalive_sec * 1000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.so_reuseport", 0),  # a second server on this port must fail loudly
        ],
    )
    server.add_generic_rpc_handlers((service.handlers(),))
    try:
        port = server.add_secure_port(args.bind, creds)
    except RuntimeError as e:
        log.error("cannot bind %s: %s", args.bind, e)
        return 1
    if not port:
        log.error("cannot bind %s", args.bind)
        return 1
    server.start()
    log.info("listening on %s (mTLS required) | allowed=%s expected/round=%d rounds=%d",
             args.bind, sorted(allowed), expected, args.rounds)
    log.info("limits: max_concurrent_rpcs=%d workers=%d streams/conn=%d msg=%dMiB "
             "round_timeout=%.0fs deadline_required=%s",
             cap, workers, args.max_concurrent_streams, args.max_message_mb, args.round_timeout,
             not args.allow_missing_deadline)

    stop = threading.Event()
    try:
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    except ValueError:  # not in main thread (e.g. when driven from a test harness)
        pass
    try:
        while not stop.wait(0.2):
            if agg.done.is_set():
                log.info("all %d rounds finished", args.rounds)
                break
    except KeyboardInterrupt:
        log.info("interrupted")
    server.stop(args.shutdown_grace).wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
