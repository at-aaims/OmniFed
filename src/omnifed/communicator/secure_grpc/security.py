# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Security helpers for the secure OmniFed gRPC communicator.

Everything that is about *who may talk to the server and how* lives here, apart from the
federated-learning logic: mTLS credential loading, caller identity taken from the verified
client certificate, server/channel option sets, concurrency planning, and deadline handling.

Only ``grpcio`` and the standard library are required. If the ``cryptography`` package is
installed, certificate/key files are also checked at start-up (expiry, key matches cert,
CN is the one this rank must present); without it those checks are skipped, and the same
mistakes are still caught later by the TLS handshake and by RegisterClient.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import stat
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import grpc

log = logging.getLogger("omnifed.secure_grpc")

# Rank-specific file locations can come from the environment, so one shared yaml works for
# every rank (each rank has a different certificate).
ENV_CA = "OMNIFED_TLS_CA"
ENV_CERT = "OMNIFED_TLS_CERT"
ENV_KEY = "OMNIFED_TLS_KEY"

DEFAULT_CLIENT_CN_PREFIX = "client-"

# grpcio does NOT return None from ServicerContext.time_remaining() when the client set no
# deadline: it returns the C-core "infinite future" converted to seconds (~9.2e18; verified
# on grpcio 1.80 and in the 1.81.1 source). Anything this large means "no deadline".
NO_DEADLINE_S = 1e12


# --------------------------------------------------------------------------------------
# TLS material
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TlsMaterial:
    """PEM bytes for one endpoint plus what could be learned about its certificate."""

    ca_pem: bytes = field(repr=False)
    cert_pem: bytes = field(repr=False)
    key_pem: bytes = field(repr=False)  # never printed
    ca_path: str = ""
    cert_path: str = ""
    key_path: str = ""
    common_name: Optional[str] = None  # None when `cryptography` is not installed
    not_after: Optional[_dt.datetime] = None


def resolve_tls_paths(
    ca_cert: Optional[str],
    cert: Optional[str],
    key: Optional[str],
    rank: int,
) -> tuple[str, str, str]:
    """Explicit argument > environment variable. ``{rank}`` in a path is substituted."""
    wanted = {
        "tls_ca_cert": (ca_cert or os.environ.get(ENV_CA), ENV_CA),
        "tls_cert": (cert or os.environ.get(ENV_CERT), ENV_CERT),
        "tls_key": (key or os.environ.get(ENV_KEY), ENV_KEY),
    }
    missing = [f"{name} (or ${env})" for name, (val, env) in wanted.items() if not val]
    if missing:
        raise ValueError(
            "secure gRPC needs TLS files, none are optional: missing " + ", ".join(missing)
        )
    out = []
    for name, (val, _env) in wanted.items():
        try:
            out.append(os.path.expanduser(str(val).format(rank=rank)))
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"{name}={val!r}: only the {{rank}} placeholder is supported in paths"
            ) from exc
    return out[0], out[1], out[2]


def _check_key_permissions(path: str, allow_loose: bool) -> None:
    if os.name != "posix":
        return
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        msg = (
            f"private key {path} is accessible by group/others (mode {mode:04o}); "
            f"fix with: chmod 600 {path}"
        )
        if allow_loose:
            log.warning("%s (continuing: tls_allow_loose_key_permissions=True)", msg)
            return
        raise PermissionError(msg)


def _read_pem(path: str, what: str, marker: bytes) -> bytes:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    with open(path, "rb") as fh:
        data = fh.read()
    if marker not in data:
        raise ValueError(f"{what} {path} is not PEM (no {marker.decode()} line)")
    return data


def _inspect_with_cryptography(cert_pem: bytes, key_pem: bytes):
    """(common_name, not_before, not_after, key_matches_cert) or None if not installed."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None

    cert = x509.load_pem_x509_certificate(cert_pem)  # first certificate in the file
    cns = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    common_name = str(cns[0].value) if cns else None

    def _utc(value_utc_attr: str, value_naive_attr: str) -> _dt.datetime:
        got = getattr(cert, value_utc_attr, None)
        if got is not None:
            return got
        return getattr(cert, value_naive_attr).replace(tzinfo=_dt.timezone.utc)

    not_before = _utc("not_valid_before_utc", "not_valid_before")
    not_after = _utc("not_valid_after_utc", "not_valid_after")

    key = serialization.load_pem_private_key(key_pem, password=None)

    def _spki(pub) -> bytes:
        return pub.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    matches = _spki(cert.public_key()) == _spki(key.public_key())
    return common_name, not_before, not_after, matches


def load_tls_material(
    ca_cert: Optional[str],
    cert: Optional[str],
    key: Optional[str],
    *,
    rank: int,
    expected_common_name: Optional[str] = None,
    allow_loose_key_permissions: bool = False,
    warn_days: int = 7,
) -> TlsMaterial:
    """Read and sanity-check this endpoint's CA bundle, certificate and private key.

    ``expected_common_name`` is what this endpoint's certificate must be named (clients
    pass the CN that maps to their rank). It is only enforced when ``cryptography`` is
    installed; otherwise the server's RegisterClient reply performs the same check.
    """
    ca_path, cert_path, key_path = resolve_tls_paths(ca_cert, cert, key, rank)
    ca_pem = _read_pem(ca_path, "CA certificate", b"-----BEGIN CERTIFICATE-----")
    cert_pem = _read_pem(cert_path, "certificate", b"-----BEGIN CERTIFICATE-----")
    _check_key_permissions(key_path, allow_loose_key_permissions)
    key_pem = _read_pem(key_path, "private key", b"-----BEGIN ")
    if b"ENCRYPTED" in key_pem:
        raise ValueError(
            f"private key {key_path} is passphrase-protected; grpc cannot use encrypted "
            "keys (store an unencrypted key with mode 600)"
        )

    common_name = not_after = None
    try:
        inspected = _inspect_with_cryptography(cert_pem, key_pem)
    except Exception as exc:  # unreadable PEM, unsupported key type, ...
        raise ValueError(f"cannot parse {cert_path} / {key_path}: {exc}") from exc

    if inspected is None:
        log.info(
            "cryptography is not installed: skipped certificate expiry / key-match / CN checks"
        )
    else:
        common_name, not_before, not_after, matches = inspected
        now = _dt.datetime.now(_dt.timezone.utc)
        if not matches:
            raise ValueError(
                f"private key {key_path} does not belong to certificate {cert_path}"
            )
        if now < not_before:
            raise ValueError(
                f"certificate {cert_path} is not valid until {not_before:%Y-%m-%d %H:%M} UTC "
                "(clock skew between nodes?)"
            )
        if now >= not_after:
            raise ValueError(f"certificate {cert_path} expired {not_after:%Y-%m-%d %H:%M} UTC")
        if not_after - now < _dt.timedelta(days=warn_days):
            log.warning(
                "certificate %s expires %s UTC (< %d days)", cert_path, f"{not_after:%Y-%m-%d}", warn_days
            )
        if expected_common_name is not None and common_name != expected_common_name:
            raise ValueError(
                f"certificate {cert_path} has CN={common_name!r} but rank {rank} must present "
                f"CN={expected_common_name!r} (wrong certificate file for this rank?)"
            )

    return TlsMaterial(
        ca_pem=ca_pem,
        cert_pem=cert_pem,
        key_pem=key_pem,
        ca_path=ca_path,
        cert_path=cert_path,
        key_path=key_path,
        common_name=common_name,
        not_after=not_after,
    )


def server_credentials(material: TlsMaterial) -> grpc.ServerCredentials:
    """mTLS: the server presents its certificate AND refuses clients without a valid one."""
    return grpc.ssl_server_credentials(
        [(material.key_pem, material.cert_pem)],
        root_certificates=material.ca_pem,
        require_client_auth=True,  # <- this is the "m" in mTLS
    )


def channel_credentials(material: TlsMaterial) -> grpc.ChannelCredentials:
    """Client side: verify the server against the CA and present our own certificate."""
    return grpc.ssl_channel_credentials(
        root_certificates=material.ca_pem,
        private_key=material.key_pem,
        certificate_chain=material.cert_pem,
    )


def tls_failure_hint(host: str) -> str:
    return (
        "TLS/connection failure. Check: (1) the server is up and reachable on that port; "
        f"(2) the server certificate's SAN contains {host!r} (gen_certs.sh -s EXTRA_SAN); "
        "(3) this client's certificate was signed by the CA the server trusts, and this "
        "client trusts the server's CA; (4) clocks are in sync and nothing is expired."
    )


# --------------------------------------------------------------------------------------
# Channel / server options
# --------------------------------------------------------------------------------------


def server_options(
    max_send_message_length: int,
    max_receive_message_length: int,
    *,
    max_concurrent_streams: Optional[int] = 16,
    keepalive_sec: Optional[float] = 60.0,
    keepalive_timeout_sec: float = 20.0,
    handshake_timeout_sec: Optional[float] = 20.0,
    idle_timeout_sec: Optional[float] = None,
) -> list[tuple[str, int]]:
    """Server channel args. ``None`` / 0 disables the optional ones."""
    opts: list[tuple[str, int]] = [
        ("grpc.max_send_message_length", int(max_send_message_length)),
        ("grpc.max_receive_message_length", int(max_receive_message_length)),
        # A second server on the same port must fail loudly instead of silently sharing it.
        ("grpc.so_reuseport", 0),
    ]
    if max_concurrent_streams:
        opts.append(("grpc.max_concurrent_streams", int(max_concurrent_streams)))
    if keepalive_sec:
        # Reaps half-open connections (a peer that died without closing its socket).
        opts.append(("grpc.keepalive_time_ms", int(keepalive_sec * 1000)))
        opts.append(("grpc.keepalive_timeout_ms", int(keepalive_timeout_sec * 1000)))
    if handshake_timeout_sec:
        # A TCP peer that never completes the TLS handshake is dropped after this long.
        opts.append(("grpc.server_handshake_timeout_ms", int(handshake_timeout_sec * 1000)))
    if idle_timeout_sec:
        opts.append(("grpc.max_connection_idle_ms", int(idle_timeout_sec * 1000)))
    return opts


def client_options(
    max_send_message_length: int,
    max_receive_message_length: int,
    *,
    server_name_override: Optional[str] = None,
) -> list[tuple[str, Any]]:
    opts: list[tuple[str, Any]] = [
        ("grpc.max_send_message_length", int(max_send_message_length)),
        ("grpc.max_receive_message_length", int(max_receive_message_length)),
    ]
    if server_name_override:
        log.warning(
            "tls_server_name_override=%r: hostname verification is bypassed for the server "
            "name. Use only for tests on 127.0.0.1; put the real hostname in the server "
            "certificate SAN instead.",
            server_name_override,
        )
        opts.append(("grpc.ssl_target_name_override", server_name_override))
    return opts


def plan_concurrency(
    world_size: int,
    max_workers: int,
    max_concurrent_rpcs: Optional[int],
) -> tuple[int, int]:
    """Return (maximum_concurrent_rpcs, thread-pool workers).

    Every client can have one RPC in flight at the same time (all of them wait for the
    aggregated result), so the cap must admit ``world_size - 1`` RPCs or a round could
    never finish. The pool always gets at least one thread per admitted RPC; otherwise
    admitted-but-queued RPCs could be starved by long-polling ones.
    """
    clients = max(int(world_size) - 1, 1)
    cap = int(max_concurrent_rpcs) if max_concurrent_rpcs else max(int(max_workers), clients + 4)
    if cap < clients:
        raise ValueError(
            f"max_concurrent_rpcs={cap} is below the {clients} clients that wait on the "
            "server at the same time; a round could never complete"
        )
    return cap, max(int(max_workers), cap)


# --------------------------------------------------------------------------------------
# Caller identity (server side)
# --------------------------------------------------------------------------------------


def identities_for_world(
    world_size: int, prefix: str = DEFAULT_CLIENT_CN_PREFIX
) -> dict[str, str]:
    """Default allowlist: certificate CN ``<prefix><rank>`` -> participant id ``"<rank>"``."""
    return {f"{prefix}{r}": str(r) for r in range(1, int(world_size))}


def normalize_identities(
    identities: Optional[Mapping[str, Any]], world_size: int, prefix: str
) -> dict[str, str]:
    """Validate an explicit {CN: rank} mapping (or build the default one)."""
    if not identities:
        return identities_for_world(world_size, prefix)
    out: dict[str, str] = {}
    for cn, rank in dict(identities).items():
        try:
            rank_i = int(rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"client identity {cn!r}: rank {rank!r} is not an integer") from exc
        if not 1 <= rank_i < int(world_size):
            raise ValueError(
                f"client identity {cn!r}: rank {rank_i} outside 1..{int(world_size) - 1}"
            )
        out[str(cn)] = str(rank_i)
    if len(set(out.values())) != len(out):
        raise ValueError("two certificate names map to the same rank; each rank needs its own")
    return out


def expected_cn_for_rank(identities: Mapping[str, str], rank: int) -> Optional[str]:
    for cn, pid in identities.items():
        if pid == str(rank):
            return cn
    return None


def tls_paths_from_dir(
    tls_dir: str, rank: int, identities: Mapping[str, str]
) -> tuple[str, str, str]:
    """File locations in the layout gen_certs.sh writes, so one shared config serves every rank::

        <tls_dir>/ca.crt
        <tls_dir>/server/server.crt|key              (rank 0)
        <tls_dir>/clients/<certificate name>.crt|key (rank >= 1)

    Each node only needs its own files in that layout (plus ca.crt); nothing else is read.
    """
    base = os.path.expanduser(str(tls_dir))
    ca = os.path.join(base, "ca.crt")
    if int(rank) == 0:
        stem = os.path.join(base, "server", "server")
    else:
        cn = expected_cn_for_rank(identities, rank)
        if cn is None:
            raise ValueError(f"no client certificate name is configured for rank {rank}")
        stem = os.path.join(base, "clients", cn)
    return ca, stem + ".crt", stem + ".key"


class IdentityPolicy:
    """Maps the verified TLS client certificate of a call to a participant id.

    The identity is read from ``ServicerContext.auth_context()`` -- data the TLS layer
    verified against the CA -- never from anything the caller wrote into a message.
    """

    def __init__(self, identities: Mapping[str, str]):
        if not identities:
            raise ValueError("IdentityPolicy needs at least one allowed certificate name")
        self._identities = dict(identities)

    @property
    def allowed_names(self) -> frozenset:
        return frozenset(self._identities)

    def resolve(self, context: grpc.ServicerContext) -> str:
        """Return the participant id, or abort the RPC (UNAUTHENTICATED / PERMISSION_DENIED)."""
        peer = context.peer()
        auth = context.auth_context() or {}
        if list(auth.get("transport_security_type") or []) != [b"ssl"]:
            log.warning("DENY no TLS identity peer=%s", peer)
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "a TLS client certificate is required")
        names = list(auth.get("x509_common_name") or [])
        if len(names) != 1:
            log.warning("DENY certificate without exactly one CN peer=%s", peer)
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "client certificate must carry exactly one CN"
            )
        cn = names[0].decode("utf-8", "replace")
        participant = self._identities.get(cn)
        if participant is None:
            log.warning("DENY identity=%r not in allowlist peer=%s", cn, peer)
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "identity not allowed")
        return participant


# --------------------------------------------------------------------------------------
# Deadlines (server side)
# --------------------------------------------------------------------------------------


def remaining_deadline(context: grpc.ServicerContext) -> Optional[float]:
    """Seconds left on the client's deadline, or None if the client set none."""
    remaining = context.time_remaining()
    if remaining is None or remaining >= NO_DEADLINE_S:
        return None
    return max(0.0, float(remaining))


def enforce_deadline(context: grpc.ServicerContext, required: bool) -> None:
    """Abort calls that carry no deadline (when required): no unbounded server-side work."""
    if required and remaining_deadline(context) is None:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "RPC deadline required: set timeout= on the call"
        )


def wait_budget(context: grpc.ServicerContext, cap_sec: float, margin_sec: float = 1.0) -> float:
    """Seconds a handler may block: the server's cap, shortened to the client's deadline."""
    remaining = remaining_deadline(context)
    if remaining is None:
        return float(cap_sec)
    return max(0.0, min(float(cap_sec), remaining - margin_sec))


# --------------------------------------------------------------------------------------
# Client-side error classification
# --------------------------------------------------------------------------------------


def is_auth_error(err: grpc.RpcError) -> bool:
    """Server said who we are is wrong. Retrying cannot help."""
    return err.code() in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED)


def is_busy_error(err: grpc.RpcError) -> bool:
    """Server admitted too many RPCs at once (maximum_concurrent_rpcs). Retry later."""
    return err.code() == grpc.StatusCode.RESOURCE_EXHAUSTED and "concurrent" in (
        (err.details() or "").lower()
    )
