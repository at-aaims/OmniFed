#!/usr/bin/env python3
"""Secure (mutual-TLS) gRPC client -- standalone test.

Each round the client generates deterministic random float32 arrays (seeded by
--seed, --index and the round), sends them to the server with an explicit RPC
deadline, and receives the sample-weighted mean over all clients.  With --verify
(default) it recomputes the expected mean locally -- possible because every client's
data is a pure function of (seed, index, round) -- and checks the server's answer.

The client authenticates with its own certificate (--cert/--key) and verifies the
server against --ca (certificate chain + hostname).  It does not trust the server's
reply either: names, dtype, shapes and finiteness are validated before use.

Run (one per terminal / node):
  python test_grpc_client.py --server <server-host>:50051 --ca certs/ca.crt \
      --cert certs/clients/client0.crt --key certs/clients/client0.key \
      --index 0 --num-clients 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import stat
import struct
import sys
import time
from math import prod

import grpc
import numpy as np

log = logging.getLogger("secure-grpc-client")

# --------------------------------------------------------------------------------------
# Wire format (keep identical in test_grpc_server.py)
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
RETRYABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.ABORTED}


class WireError(ValueError):
    """Malformed or unacceptable frame."""


def parse_shapes(text: str) -> dict[str, tuple[int, ...]]:
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


def spec_nbytes(spec) -> int:
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


def decode_frame(buf: bytes, spec) -> tuple[dict, dict]:
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


# --------------------------------------------------------------------------------------
# Data generation + local verification
# --------------------------------------------------------------------------------------
def samples_for(index: int, base_samples: int) -> int:
    return base_samples * (index + 1)


def make_arrays(spec, seed: int, index: int, rnd: int) -> dict:
    rng = np.random.default_rng([seed, index, rnd])
    return {k: rng.standard_normal(shape).astype(np.float32) for k, shape in spec.items()}


def expected_mean(spec, seed: int, num_clients: int, base_samples: int, rnd: int):
    """What the server should return: sum(n_j * x_j) / sum(n_j) over all clients j."""
    acc = {k: np.zeros(shape, dtype=np.float64) for k, shape in spec.items()}
    total = 0
    for j in range(num_clients):
        n = samples_for(j, base_samples)
        for k, arr in make_arrays(spec, seed, j, rnd).items():
            acc[k] += arr.astype(np.float64) * n
        total += n
    return {k: v / total for k, v in acc.items()}, total


def max_abs_err(got: dict, want: dict) -> float:
    return max(float(np.max(np.abs(got[k].astype(np.float64) - want[k]))) for k in want)


# --------------------------------------------------------------------------------------
# Transport
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


def open_channel(args) -> grpc.Channel:
    _warn_if_key_open(args.key)
    creds = grpc.ssl_channel_credentials(
        root_certificates=_read(args.ca),          # verify the server against OUR CA
        private_key=_read(args.key),               # present OUR identity
        certificate_chain=_read(args.cert),
    )
    max_msg = args.max_message_mb * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", max_msg),
        ("grpc.max_send_message_length", max_msg),
        ("grpc.enable_retries", 0),  # retries are explicit below (the server is idempotent)
    ]
    if args.server_name_override:
        log.warning("--server-name-override=%s: hostname check uses this name instead of the "
                    "one you connect to (CA check still applies). Testing only.",
                    args.server_name_override)
        options.append(("grpc.ssl_target_name_override", args.server_name_override))
    return grpc.secure_channel(args.server, creds, options=options)


def call_with_retry(method, request: bytes, deadline: float, retries: int, backoff: float, what: str) -> bytes:
    for attempt in range(retries + 1):
        try:
            return method(request, timeout=deadline)  # explicit per-RPC deadline
        except grpc.RpcError as e:
            if e.code() not in RETRYABLE or attempt == retries:
                raise
            wait = min(backoff * 2**attempt, 30.0)
            log.warning("%s: %s (%s); retry %d/%d in %.1fs", what, e.code().name,
                        (e.details() or "")[:120], attempt + 1, retries, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mTLS gRPC client (standalone test)")
    p.add_argument("--server", required=True, help="host:port; host must appear in the server cert SAN")
    p.add_argument("--ca", required=True, help="CA certificate used to verify the server")
    p.add_argument("--cert", required=True, help="this client's certificate (PEM)")
    p.add_argument("--key", required=True, help="this client's private key (PEM)")
    p.add_argument("--index", type=int, required=True, help="client index (selects data + sample count)")
    p.add_argument("--num-clients", type=int, required=True, help="total clients (for --verify)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="name:AxB,... (must match server)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base-samples", type=int, default=100, help="num_samples = base * (index+1)")
    p.add_argument("--rpc-deadline", type=float, default=120.0,
                   help="deadline in seconds for EACH Aggregate/WhoAmI attempt [120]")
    p.add_argument("--connect-timeout", type=float, default=30.0, help="seconds to wait for mTLS channel [30]")
    p.add_argument("--max-retries", type=int, default=3, help="retries for UNAVAILABLE/DEADLINE/ABORTED [3]")
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--max-message-mb", type=int, default=64)
    p.add_argument("--server-name-override", default="",
                   help="TESTING ONLY: hostname to verify the server cert against")
    p.add_argument("--no-verify", dest="verify", action="store_false",
                   help="skip recomputing the expected mean locally")
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
    if not (0 <= args.index < args.num_clients):
        log.error("--index must be in [0, --num-clients)")
        return 2

    channel = open_channel(args)
    host = args.server.rsplit(":", 1)[0]
    try:
        grpc.channel_ready_future(channel).result(timeout=args.connect_timeout)
    except grpc.FutureTimeoutError:
        log.error(
            "mTLS channel to %s not ready after %.0fs. Typical causes:\n"
            "  - server not running, wrong port, or port blocked (try: nc -zv %s %s)\n"
            "  - '%s' is not in the server certificate SAN (regenerate with gen_certs.sh -s)\n"
            "  - server cert or this client's cert not signed by the CA given in --ca, or expired\n"
            "  - re-run with GRPC_VERBOSITY=debug GRPC_TRACE=handshaker,tsi for details",
            args.server, args.connect_timeout, host, args.server.rsplit(":", 1)[-1], host)
        return 3

    who = channel.unary_unary(f"/{SERVICE}/WhoAmI")
    aggregate = channel.unary_unary(f"/{SERVICE}/Aggregate")
    n_i = samples_for(args.index, args.base_samples)
    try:
        info = json.loads(call_with_retry(who, b"", args.rpc_deadline, args.max_retries,
                                          args.retry_backoff, "WhoAmI"))
        log.info("server sees me as identity=%r (cert sha256 %s...) from %s",
                 info.get("identity"), str(info.get("cert_sha256"))[:16], info.get("peer"))
        ok = True
        for rnd in range(args.rounds):
            mine = make_arrays(spec, args.seed, args.index, rnd)
            req = encode_frame({"round": rnd, "num_samples": n_i}, mine)
            t0 = time.perf_counter()
            raw = call_with_retry(aggregate, req, args.rpc_deadline, args.max_retries,
                                  args.retry_backoff, f"Aggregate r{rnd}")
            dt = time.perf_counter() - t0
            meta, got = decode_frame(raw, spec)
            if meta.get("round") != rnd:
                raise WireError(f"server answered round {meta.get('round')!r}, expected {rnd}")
            line = (f"round {rnd}: participants={meta.get('participants')} "
                    f"total_samples={meta.get('total_samples')} sent={len(req)/1e6:.2f}MB "
                    f"rtt={dt:.2f}s sha256={hashlib.sha256(raw).hexdigest()[:12]}")
            if args.verify:
                want, total = expected_mean(spec, args.seed, args.num_clients, args.base_samples, rnd)
                err = max_abs_err(got, want)
                good = (err < 1e-5 and meta.get("participants") == args.num_clients
                        and meta.get("total_samples") == total)
                ok &= good
                line += f" max|err|={err:.2e} {'OK' if good else 'MISMATCH'}"
            log.info(line)
        log.info("done: %s", "all rounds verified" if ok and args.verify else "finished")
        return 0 if ok else 4
    except grpc.RpcError as e:
        log.error("RPC failed: %s: %s", e.code().name, e.details())
        return 5
    except WireError as e:
        log.error("rejected server response: %s", e)
        return 6
    finally:
        channel.close()


if __name__ == "__main__":
    sys.exit(main())
