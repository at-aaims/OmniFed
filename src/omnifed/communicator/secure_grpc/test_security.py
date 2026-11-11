#!/usr/bin/env python3
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

"""Self-test for security.py using real gRPC servers on 127.0.0.1 (needs grpcio + openssl).

Needs neither torch nor the generated *_pb2 files, so it can run before you have run protoc:

    cd src/omnifed/communicator/secure_grpc && python test_security.py

It makes a throw-away PKI with gen_certs.sh in a temp directory and checks, against real
mTLS connections: identity comes from the certificate, an unlisted-but-valid certificate is
refused, a foreign CA / missing client certificate / plaintext are refused, hostname
verification is active, deadlines are enforced, the concurrent-RPC cap rejects the excess,
a stalled TLS handshake is dropped, and a second server cannot silently share the port.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# The C-core logs every refused handshake to stderr; those are the *expected* outcomes here.
# Set GRPC_VERBOSITY=ERROR (or unset this) to see them.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")

import grpc  # noqa: E402  (after HERE so a stray local grpc.py would be obvious)


def _load_security():
    spec = importlib.util.spec_from_file_location("secure_grpc_security", os.path.join(HERE, "security.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sec = _load_security()
logging.getLogger("omnifed.secure_grpc").setLevel(logging.CRITICAL)  # DENY / override warnings are expected

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def make_pki(root: str, names: list[str]) -> str:
    out = os.path.join(root, "pki")
    subprocess.run(
        ["bash", os.path.join(HERE, "gen_certs.sh"), "-o", out, *names],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def material(pki: str, who: str, rank: int = 1, expected_cn=None):
    if who == "server":
        cert, key = f"{pki}/server/server.crt", f"{pki}/server/server.key"
    else:
        cert, key = f"{pki}/clients/{who}.crt", f"{pki}/clients/{who}.key"
    return sec.load_tls_material(f"{pki}/ca.crt", cert, key, rank=rank, expected_common_name=expected_cn)


# ---- a tiny service whose handlers use the helpers exactly like the real servicer does ----
def make_server(pki, policy, *, cap=8, workers=8, options=None, require_deadline=True, port=0):
    ident = lambda b: b  # noqa: E731

    def who(req, ctx):
        sec.enforce_deadline(ctx, require_deadline)
        return sec.IdentityPolicy.resolve(policy, ctx).encode()

    def budget(req, ctx):
        sec.IdentityPolicy.resolve(policy, ctx)
        return f"{sec.wait_budget(ctx, 30.0):.2f}".encode()

    def slow(req, ctx):
        sec.IdentityPolicy.resolve(policy, ctx)
        time.sleep(1.0)
        return b"ok"

    handlers = grpc.method_handlers_generic_handler(
        "t.T",
        {
            "Who": grpc.unary_unary_rpc_method_handler(who, ident, ident),
            "Budget": grpc.unary_unary_rpc_method_handler(budget, ident, ident),
            "Slow": grpc.unary_unary_rpc_method_handler(slow, ident, ident),
        },
    )
    server = grpc.server(
        cf.ThreadPoolExecutor(max_workers=workers),
        maximum_concurrent_rpcs=cap,
        options=options or sec.server_options(1 << 20, 1 << 20),
        handlers=[handlers],
    )
    bound = server.add_secure_port(f"127.0.0.1:{port}", sec.server_credentials(material(pki, "server")))
    server.start()
    return server, bound


def call(channel, method, timeout=5.0):
    return channel.unary_unary(f"/t.T/{method}", request_serializer=lambda b: b, response_deserializer=lambda b: b)(
        b"", timeout=timeout
    )


def secure_channel(port, creds, override="localhost"):
    return grpc.secure_channel(f"127.0.0.1:{port}", creds, options=sec.client_options(1 << 20, 1 << 20, server_name_override=override))


def code_of(fn):
    try:
        fn()
        return None
    except grpc.RpcError as e:
        return e


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="secure_grpc_test_")
    try:
        return run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp: str) -> int:
    print("PKI ...")
    pki = make_pki(tmp, ["client-1", "client-2", "client-9"])
    rogue = os.path.join(tmp, "rogue")
    os.makedirs(rogue)
    rogue_pki = make_pki(rogue, ["client-1"])

    print("\n[1] path resolution and file checks")
    ca, ce, ke = sec.resolve_tls_paths("a/{rank}/ca", "b-{rank}.crt", "c.key", 3)
    check("{rank} is substituted", (ca, ce, ke) == ("a/3/ca", "b-3.crt", "c.key"), str((ca, ce, ke)))
    for k in (sec.ENV_CA, sec.ENV_CERT, sec.ENV_KEY):
        os.environ.pop(k, None)
    try:
        sec.resolve_tls_paths(None, "x", "y", 0)
        check("missing tls_ca_cert is an error", False)
    except ValueError as e:
        check("missing tls_ca_cert is an error", "tls_ca_cert" in str(e))
    os.environ[sec.ENV_CA] = "/env/ca.crt"
    check("environment fallback", sec.resolve_tls_paths(None, "x", "y", 0)[0] == "/env/ca.crt")
    os.environ.pop(sec.ENV_CA)
    try:
        sec.resolve_tls_paths("a{oops}", "x", "y", 0)
        check("unknown placeholder rejected", False)
    except ValueError:
        check("unknown placeholder rejected", True)

    keyfile = f"{pki}/clients/client-1.key"
    check("a mode-600 key is accepted", material(pki, "client-1").key_pem.startswith(b"-----BEGIN"))
    if os.name == "posix":
        loose = os.path.join(tmp, "loose.key")
        shutil.copy(keyfile, loose)
        os.chmod(loose, 0o644)
        try:
            sec.load_tls_material(f"{pki}/ca.crt", f"{pki}/clients/client-1.crt", loose, rank=1)
            check("group/world-readable key refused", False)
        except PermissionError:
            check("group/world-readable key refused", True)
        check(
            "...unless explicitly allowed",
            sec.load_tls_material(f"{pki}/ca.crt", f"{pki}/clients/client-1.crt", loose, rank=1,
                                  allow_loose_key_permissions=True) is not None,
        )
    enc = os.path.join(tmp, "enc.key")
    with open(enc, "w") as fh:
        fh.write("-----BEGIN ENCRYPTED PRIVATE KEY-----\nAAAA\n-----END ENCRYPTED PRIVATE KEY-----\n")
    os.chmod(enc, 0o600)
    try:
        sec.load_tls_material(f"{pki}/ca.crt", f"{pki}/clients/client-1.crt", enc, rank=1)
        check("passphrase-protected key refused with a clear message", False)
    except ValueError as e:
        check("passphrase-protected key refused with a clear message", "passphrase" in str(e))
    try:
        sec.load_tls_material(f"{pki}/nope.crt", f"{pki}/clients/client-1.crt", keyfile, rank=1)
        check("missing file named in the error", False)
    except FileNotFoundError as e:
        check("missing file named in the error", "nope.crt" in str(e))
    try:
        sec.load_tls_material(f"{pki}/ca.key", f"{pki}/clients/client-1.crt", keyfile, rank=1)
        check("a key file given as the CA is rejected as not-a-certificate", False)
    except ValueError:
        check("a key file given as the CA is rejected as not-a-certificate", True)

    m = material(pki, "client-1", expected_cn="client-1")
    try:
        import cryptography  # noqa: F401

        have_crypto = True
    except ImportError:
        have_crypto = False
    if have_crypto:
        check("CN read from the certificate", m.common_name == "client-1", str(m.common_name))
        try:
            material(pki, "client-1", rank=2, expected_cn="client-2")
            check("rank/certificate CN mismatch caught before connecting", False)
        except ValueError as e:
            check("rank/certificate CN mismatch caught before connecting", "client-2" in str(e))
        try:
            sec.load_tls_material(f"{pki}/ca.crt", f"{pki}/clients/client-1.crt", f"{pki}/clients/client-2.key", rank=1)
            check("key that does not match the certificate caught", False)
        except ValueError as e:
            check("key that does not match the certificate caught", "does not belong" in str(e))
    else:
        print("  [SKIP] cryptography not installed: expiry / key-match / CN pre-checks not exercised")

    print("\n[2] identities and concurrency planning")
    ids = sec.identities_for_world(4)
    check("default allowlist is client-1..client-3", ids == {"client-1": "1", "client-2": "2", "client-3": "3"}, str(ids))
    check("explicit mapping normalised", sec.normalize_identities({"alice": 2, "bob": "1"}, 3, "client-") == {"alice": "2", "bob": "1"})
    for bad, why in (({"a": 0}, "rank 0 is the server"), ({"a": 3}, "rank outside world"), ({"a": 1, "b": 1}, "duplicate rank"), ({"a": "x"}, "not an int")):
        try:
            sec.normalize_identities(bad, 3, "client-")
            check(f"rejects {why}", False)
        except ValueError:
            check(f"rejects {why}", True)
    check("expected CN for a rank", sec.expected_cn_for_rank(ids, 2) == "client-2")
    check("plan: default admits all waiting clients", sec.plan_concurrency(21, 10, None) == (24, 24), str(sec.plan_concurrency(21, 10, None)))
    check("plan: pool never smaller than the cap", sec.plan_concurrency(3, 4, 9) == (9, 9))
    try:
        sec.plan_concurrency(10, 10, 3)
        check("plan: cap below the number of waiting clients refused", False)
    except ValueError:
        check("plan: cap below the number of waiting clients refused", True)

    policy = sec.IdentityPolicy(sec.identities_for_world(3))

    class FakeCtx:
        def __init__(self, auth):
            self._auth, self.aborted = auth, None

        def peer(self):
            return "ipv4:fake"

        def auth_context(self):
            return self._auth

        def abort(self, code, details):
            self.aborted = code
            raise RuntimeError("aborted")

    for label, auth, want in (
        ("no TLS identity", {}, grpc.StatusCode.UNAUTHENTICATED),
        ("two CNs", {"transport_security_type": [b"ssl"], "x509_common_name": [b"client-1", b"client-2"]}, grpc.StatusCode.UNAUTHENTICATED),
        ("unlisted CN", {"transport_security_type": [b"ssl"], "x509_common_name": [b"mallory"]}, grpc.StatusCode.PERMISSION_DENIED),
    ):
        ctx = FakeCtx(auth)
        try:
            policy.resolve(ctx)
        except RuntimeError:
            pass
        check(f"policy aborts: {label}", ctx.aborted == want, str(ctx.aborted))
    check("policy maps CN to participant id", policy.resolve(FakeCtx({"transport_security_type": [b"ssl"], "x509_common_name": [b"client-2"]})) == "2")

    print("\n[3] real mTLS connections")
    server, port = make_server(pki, policy)
    try:
        for who, rank in (("client-1", 1), ("client-2", 2)):
            with secure_channel(port, sec.channel_credentials(material(pki, who, rank))) as ch:
                got = call(ch, "Who").decode()
                check(f"{who} is identified as participant {rank} from its certificate", got == str(rank), got)

        with secure_channel(port, sec.channel_credentials(material(pki, "client-9", 1))) as ch:
            e = code_of(lambda: call(ch, "Who"))
            check("valid certificate but not on the allowlist -> PERMISSION_DENIED",
                  e is not None and e.code() == grpc.StatusCode.PERMISSION_DENIED, str(e and e.code()))
            check("...and that is not retryable", e is not None and sec.is_auth_error(e))

        rogue_m = sec.load_tls_material(f"{rogue_pki}/ca.crt", f"{rogue_pki}/clients/client-1.crt", f"{rogue_pki}/clients/client-1.key", rank=1)
        with secure_channel(port, sec.channel_credentials(rogue_m)) as ch:
            e = code_of(lambda: call(ch, "Who", timeout=4))
            check("client certificate from a foreign CA refused", e is not None and e.code() == grpc.StatusCode.UNAVAILABLE, str(e and e.code()))

        good = material(pki, "client-1", 1)
        no_cert = grpc.ssl_channel_credentials(root_certificates=good.ca_pem)
        with secure_channel(port, no_cert) as ch:
            e = code_of(lambda: call(ch, "Who", timeout=4))
            check("client without any certificate refused (mTLS, not just TLS)", e is not None and e.code() == grpc.StatusCode.UNAVAILABLE, str(e and e.code()))

        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            e = code_of(lambda: call(ch, "Who", timeout=4))
            check("plaintext client refused", e is not None and e.code() == grpc.StatusCode.UNAVAILABLE, str(e and e.code()))

        with secure_channel(port, sec.channel_credentials(good), override="evil.example") as ch:
            e = code_of(lambda: call(ch, "Who", timeout=4))
            check("hostname verification is active (name not in server SAN refused)",
                  e is not None and e.code() == grpc.StatusCode.UNAVAILABLE, str(e and e.code()))

        wrong_ca = sec.load_tls_material(f"{rogue_pki}/ca.crt", f"{pki}/clients/client-1.crt", f"{pki}/clients/client-1.key", rank=1)
        with secure_channel(port, sec.channel_credentials(wrong_ca)) as ch:
            e = code_of(lambda: call(ch, "Who", timeout=4))
            check("server certificate from a CA the client does not trust refused", e is not None and e.code() == grpc.StatusCode.UNAVAILABLE, str(e and e.code()))

        with secure_channel(port, sec.channel_credentials(good)) as ch:
            check("deadline present -> allowed", call(ch, "Who", timeout=5) == b"1")
            e = code_of(lambda: ch.unary_unary("/t.T/Who", request_serializer=lambda b: b, response_deserializer=lambda b: b)(b""))
            check("no deadline -> INVALID_ARGUMENT", e is not None and e.code() == grpc.StatusCode.INVALID_ARGUMENT, str(e and e.code()))
            b = float(call(ch, "Budget", timeout=5).decode())
            check("wait budget is shortened to the client deadline (5s - 1s margin)", 3.0 <= b <= 4.1, str(b))
            b = float(ch.unary_unary("/t.T/Budget", request_serializer=lambda b: b, response_deserializer=lambda b: b)(b"").decode())
            check("without a client deadline the budget is the server cap", b == 30.0, str(b))
    finally:
        server.stop(0)

    print("\n[4] concurrent-RPC cap")
    server, port = make_server(pki, policy, cap=2, workers=2)
    try:
        with secure_channel(port, sec.channel_credentials(good)) as ch:
            call(ch, "Who")  # warm the connection so all four start together
            results: list = []
            lock = threading.Lock()

            def one():
                e = code_of(lambda: call(ch, "Slow", timeout=15))
                with lock:
                    results.append(e)

            ts = [threading.Thread(target=one) for _ in range(4)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            ok = sum(1 for r in results if r is None)
            busy = [r for r in results if r is not None and r.code() == grpc.StatusCode.RESOURCE_EXHAUSTED]
            check("2 admitted, 2 rejected with RESOURCE_EXHAUSTED", ok == 2 and len(busy) == 2, f"ok={ok} busy={len(busy)} all={[r and r.code() for r in results]}")
            check("the rejection is recognised as 'busy', not 'message too large'", bool(busy) and all(sec.is_busy_error(r) for r in busy),
                  str([r.details() for r in busy]))
    finally:
        server.stop(0)

    print("\n[5] stalled handshake and duplicate bind")
    opts = sec.server_options(1 << 20, 1 << 20, handshake_timeout_sec=2)
    server, port = make_server(pki, policy, options=opts)
    try:
        s = socket.create_connection(("127.0.0.1", port))
        s.settimeout(12)
        t0 = time.time()
        try:
            s.recv(1)
            closed = True
        except ConnectionError:
            closed = True
        except socket.timeout:
            closed = False
        dt = time.time() - t0
        s.close()
        check("connection that never starts TLS is dropped (~2 s)", closed and 1.0 <= dt <= 8.0, f"closed={closed} after {dt:.1f}s")

        e = None
        try:
            server2, bound2 = make_server(pki, policy, port=port)
            server2.stop(0)
            e = bound2
        except RuntimeError as exc:
            e = exc
        check("a second server on the same port fails instead of sharing it", isinstance(e, RuntimeError) or e == 0, repr(e))
    finally:
        server.stop(0)

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
