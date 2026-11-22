# Secure gRPC (mTLS)

This folder has two parts. **Part 1** (this top section down to "How this maps to the review findings") is the
standalone test you ran on Frontier. **Part 2**, at the end, is the secure OmniFed communicator
(`SecureGrpcCommunicator`) with the protoc instructions.

## Part 1: standalone mTLS test

A self-contained prototype of what OmniFed's gRPC aggregation could look like with
mutual TLS, caller identity from the certificate, explicit deadlines and bounded
concurrency. These scripts do **not** import OmniFed, protobuf or torch (only `grpcio` + `numpy`),
and do not touch the existing communicators.

| File | Purpose |
|---|---|
| `gen_certs.sh` | Throw-away CA, one server certificate, one certificate per client (openssl only) |
| `test_grpc_server.py` | mTLS server; averages the clients' arrays each round |
| `test_grpc_client.py` | mTLS client; sends random arrays, checks the returned average |

## Does mTLS give every client its own certificate?

Yes. Plain TLS only proves the *server's* identity to the client. In mutual TLS:

* a **CA** (here `ca.crt`) signs the server certificate and one certificate per client;
* the server presents its certificate; the client checks the CA signature **and** that the
  hostname it dialled is in the certificate's SAN list;
* the client presents *its own* certificate; the server checks the CA signature and
  reads the CN (e.g. `client0`) via `context.auth_context()["x509_common_name"]`;
* a client without a certificate signed by that CA never gets past the handshake, so no
  RPC handler runs for it.

The CN is then the caller's identity, and `--allowed-clients` is the allowlist.
In a real multi-site deployment each facility generates its own key + CSR and sends only
the CSR to the CA; private keys never leave their owner. `gen_certs.sh` makes everything in
one place purely for testing.

## What each certificate and key file is

`gen_certs.sh` writes `ca.crt`, `ca.key`, `server/server.crt`, `server/server.key` and, for each
client name, `clients/<name>.crt` and `clients/<name>.key`. A `.crt` is a public certificate: it
can be shared, and it is sent over the wire during the TLS handshake. A `.key` is the matching
private key: it proves that the holder owns the certificate and must never leave the machine
that owns it.

| File | What it is | Secret? | Who holds it |
|---|---|---|---|
| `ca.crt` | The CA's public certificate, the trust anchor. Each side uses it to check that the other side's certificate was signed by this CA | no | server and every client |
| `ca.key` | The CA's private key. Whoever has it can mint a valid identity for anyone | **yes** | only the CA operator; no running process needs it |
| `server/server.crt` | The server's identity: signed by the CA, lists the hostnames/IPs clients may dial (SAN), usable only as a server certificate (`serverAuth`) | no | the server; each client receives it during the handshake |
| `server/server.key` | Proves the server owns `server.crt` | **yes** | the server only |
| `clients/<name>.crt` | One client's identity: signed by the CA, CN = `<name>` (the identity the server sees and checks against `--allowed-clients`), usable only as a client certificate (`clientAuth`) | no | that client; the server receives it during the handshake |
| `clients/<name>.key` | Proves that client owns `<name>.crt` | **yes** | that client only |

What each process needs at run time:

* **Server:** `ca.crt` (to verify client certificates), `server.crt`, `server.key`
* **Client `<name>`:** `ca.crt` (to verify the server), `<name>.crt`, `<name>.key`

No process needs `ca.key`, and no node needs another party's `.key`. The server does not need a
copy of any client's `.crt` either: it receives it in the handshake, verifies it against
`ca.crt`, and reads the identity from it. Never commit `*.key` files to git.

In this test `gen_certs.sh` creates all of these in one directory for convenience, so on a shared
filesystem every process can read every file, including the other clients' keys.

## Running it (server + 2 clients on three login nodes)

Keep the certificates **outside the git checkout**, they contain private keys.

```bash
# --- Terminal 1: the node that will run the server -------------------------------
cd src/omnifed/communicator/secure_grpc
export CERT_DIR="$(pwd)/secure_grpc_certs
bash gen_certs.sh -o $CERT_DIR client0 client1       # SAN defaults include `hostname -f`
python test_grpc_server.py \
    --ca   $CERT_DIR/ca.crt \
    --cert $CERT_DIR/server/server.crt \
    --key  $CERT_DIR/server/server.key \
    --allowed-clients client0,client1 --rounds 3
hostname -f                                                        # clients must dial exactly this name

# --- Terminal 2 (another login node) ---------------------------------------------
python test_grpc_client.py --server <server-hostname>:50051 \
    --ca $CERT_DIR/ca.crt \
    --cert $CERT_DIR/clients/client0.crt --key $CERT_DIR/clients/client0.key \
    --index 0 --num-clients 2 --rounds 3

# --- Terminal 3 (a third login node) ---------------------------------------------
python test_grpc_client.py --server <server-hostname>:50051 \
    --ca $CERT_DIR/ca.crt \
    --cert $CERT_DIR/clients/client1.crt --key $CERT_DIR/clients/client1.key \
    --index 1 --num-clients 2 --rounds 3
```

Start order does not matter: clients wait up to `--connect-timeout` for the server.
Success looks like `round r: participants=2 total_samples=300 ... max|err|=... OK` on both
clients with identical `sha256=` values, and `all rounds verified` at the end. The server
logs each submission with the identity taken from the certificate and exits after the last round.

If the certificates are on a shared filesystem that is fine for this test, but note that
every client then *can read* every other client's key; do not do that for a real run.
To use different hostnames or IPs, add them with `gen_certs.sh -s "DNS:other.name,IP:1.2.3.4"`.
Single-node smoke test: use `--bind 127.0.0.1:50051` on the server and `--server localhost:50051`.

### Options you asked to control

| Concern | Flag | Where | Default |
|---|---|---|---|
| Per-RPC deadline | `--rpc-deadline` (also `--max-retries`, `--retry-backoff`, `--connect-timeout`) | client | 120 s per attempt, 3 retries |
| Server-side cap on how long a handler may wait | `--round-timeout` | server | 300 s |
| Reject RPCs that carry no deadline | on by default; `--allow-missing-deadline` disables | server | strict |
| Max in-flight RPCs (excess get `RESOURCE_EXHAUSTED`) | `--max-concurrent-rpcs` | server | clients + 4 |
| Worker threads | `--max-workers` (never below the RPC cap) | server | = RPC cap |
| HTTP/2 streams per connection | `--max-concurrent-streams` | server | 8 |
| Message size | `--max-message-mb` | both | 64 MiB |
| Idle connection reaping / keepalive | `--idle-timeout-sec`, `--keepalive-sec` | server | 600 s / 30 s |
| Who may take part | `--allowed-clients`, `--num-clients` | server | required |
| Tensor spec (names/shapes) | `--shapes name:AxB,...` | both, must match | 3 small layers |

gRPC has no "maximum connections" option. In-flight work is bounded by
`--max-concurrent-rpcs`; connections are bounded in practice by mTLS (only certified peers
finish the handshake) plus idle-connection reaping.

## What was added compared to the default gRPC implementation

"Default" = `src/omnifed/communicator/grpc*.py` (classic) and
`src/omnifed/hierarchical/communicator/global_grpc*.py` (what hierarchical leaders use).

| Area | Default implementation | This prototype |
|---|---|---|
| Transport | `add_insecure_port("[::]:port")` / `grpc.insecure_channel`: plaintext | `add_secure_port` + `secure_channel`, TLS only, **no insecure port** |
| Server authenticity | none; a client talks to whatever answers | client verifies CA chain **and hostname/SAN** |
| Client authentication | none | `require_client_auth=True`: client must present a certificate signed by the CA |
| Caller identity | `client_id` string inside the request (`str(rank)` in the classic path), self-asserted | request has **no id field**; identity = certificate CN from `context.auth_context()` |
| Authorization | `registered_clients` is only counted, never checked | allowlist (`--allowed-clients`); every handler is wrapped so none can skip authentication |
| Cert purpose separation | n/a | server cert is `serverAuth` only, client certs `clientAuth` only; a server cert cannot be used as a client identity |
| Round control | hierarchical server: a larger `round_number` from any caller wipes the accumulator and starts a new round | the **server owns** the round counter; a request for any other round is `FAILED_PRECONDITION` |
| Duplicates / retries | hierarchical: no dedupe (fake or repeated updates complete a round). Classic: duplicate id's tensors are dropped but its `num_samples` is still added to `total_samples` | one contribution per identity per round, repeats are idempotent (nothing is double counted); a lost reply is replayed for the previous round |
| `num_samples` | any int32 (0 gives inf/NaN in the average; huge values dominate it) | must be in `[1, --max-samples]` |
| Tensor validation | dtype whitelist only; names/shapes not checked against the model; `copy_` on the client broadcasts mismatched shapes; no finiteness check; TopK allocates from a caller-supplied `original_shape` | names, dtype and shapes must equal `--shapes` **before** any array is built; NaN/Inf rejected; trailing/truncated bytes rejected |
| Reply handling on the client | copied into the model as received | client validates round, participants, names, shapes, dtype, finiteness (and can recompute the expected mean) |
| Result delivery | separate `GetAggregationResult(client_id)`; any caller who knows a participant id can fetch it and mark it delivered | result is returned **in the same authenticated RPC**; there is nothing to fetch by id |
| Waiting | classic: `event.wait()` with no timeout. Hierarchical leader: `get_averaged_model` loops forever | handler waits `min(--round-timeout, client deadline)`; cancelled RPCs release their worker immediately (termination callback); stale partial rounds are dropped (`ABORTED`) so old sums never mix into a new round |
| Client deadlines | none on submit; retry loops with fixed counts | explicit `timeout=` on every RPC (`--rpc-deadline`), bounded retries with backoff on `UNAVAILABLE`/`DEADLINE_EXCEEDED`/`ABORTED` |
| Concurrency | 10 workers, no RPC cap | `maximum_concurrent_rpcs`, workers never below the cap, streams per connection, startup refuses a cap below clients-per-round (a round could never complete) |
| Message size | 2 GiB (`INT32_MAX`) on both sides | explicit cap (64 MiB default); startup refuses a cap smaller than the payload |
| Idle / dead peers | none | idle-connection reaping + server keepalive pings |
| Port binding | `[::]`, SO_REUSEPORT on by default (verified: a second server with default options binds the same port silently) | `--bind` explicit; `grpc.so_reuseport=0`, so a second server on the port fails loudly |
| Logging | prints the caller-supplied id verbatim | allowlisted names only; `DENY`/`REJECT` lines with peer address; unknown identities logged with `%r` (control characters escaped) |
| Key hygiene | n/a | both sides warn if the private key is group/world readable |
| Wire format | protobuf (`grpc.proto`) | JSON header + raw float32 bytes via generic handlers (keeps this test free of `protoc`/generated-code version issues; no pickle). Not a security difference by itself |

### How this maps to the review findings

| Finding | Status in this prototype |
|---|---|
| 1. Plaintext transport | Fixed (TLS 1.2/1.3 only, see "How it was tested") |
| 2. No authentication | Fixed (mTLS + CN allowlist) |
| 3. Hierarchical server: caller-controlled round / duplicates / `number_samples` | Fixed by design here; **still needs porting** into `global_grpc_server.py` |
| 4. Classic server identity mix-ups (spoofing, result theft, `total_samples` inflation) | Fixed |
| 5. Resource exhaustion | **Mitigated, not eliminated**: caps, bounded waits, shape-bounded decoding; a *certified* client can still send up to the message cap |
| 6. Clients trust the server blindly | Server is authenticated by certificate; client also validates replies |
| 7. No robust aggregation | **Not addressed.** Only finiteness and `num_samples` bounds. A certified but malicious client can still skew the average |

---

# Part 2: the secure OmniFed communicator (`SecureGrpcCommunicator`)

The same centralized broadcast / aggregate protocol as `GrpcCommunicator` (rank 0 serves, ranks
1..N-1 are clients, the server also contributes its own tensors), served over mutual TLS. It has its own proto, server, client and communicator, and it does not edit
or subclass anything in `src/omnifed/communicator/`.

| File | What it is                                                                                                                                                  |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `secure_grpc.proto` | Service `SecureGrpcServer` in package `src.omnifed.communicator.secure_grpc`. No `client_id` on any request.                                                |
| `secure_grpc_pb2.py`, `secure_grpc_pb2_grpc.py` | **Generated via `protoc`** (next section).                                                                                                                  |
| `secure_grpc_communicator.py` | `SecureGrpcCommunicator` (similar to `GrpcCommunicator`): secure port on rank 0, secure channel on clients                                                  |
| `secure_grpc_server.py` | `SecureGrpcServer` servicer (similar to `GrpcServer`): every RPC starts by authenticating the caller                                                           |
| `secure_grpc_client.py` | `SecureGrpcClient` (similar to `GrpcClient`): mTLS channel, deadline on every RPC, no retry on auth refusals                                                   |
| `security.py` | Everything about *who may talk and how*: credentials, certificate identity, server options, deadlines, concurrency planning                                 |
| `secure_codec.py` | Tensor encoder for this proto's messages (the existing one is hard-wired to `grpc_pb2`); wire-compat check                                                  |
| `smoke_secure_grpc.py` | Three-process smoke test with real torch (after protoc)                                                                                                  |
| `test_security.py` | Self-test of `security.py` over real mTLS (needs only `grpcio` + `openssl`, no protoc): `cd src/omnifed/communicator/secure_grpc && python test_security.py` |
| `secure_grpc.example.yaml` | Hydra config to copy into `conf/communicator/`                                                                                                              |

The files are deliberately **not** named `grpc.py` / `grpc_pb2.py`: next to a script, a module called
`grpc` shadows the grpcio package. For the same reason run Python from the repo root, never from
inside `src/omnifed/communicator/` (the existing `grpc.py` lives there).

## 1. Generate the protobuf code (protoc)

From the repository root, in the same conda environment you run OmniFed from (the repo pins
`grpcio-tools==1.81.0` and `protobuf==6.33.6`; generate with the tools you will run with, because the
generated file records the protobuf version it was built for):

```bash
cd /Users/ssq/Documents/github/ornl_projects/OmniFed-main

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
    ./src/omnifed/communicator/secure_grpc/secure_grpc.proto
```

This is exactly how `main.sh` builds `grpc.proto`. Generate 2 files in `secure_grpc` module:

```
src/omnifed/communicator/secure_grpc/secure_grpc_pb2.py
src/omnifed/communicator/secure_grpc/secure_grpc_pb2_grpc.py
```

Check that they load (still from the repo root):

```bash
ls src/omnifed/communicator/secure_grpc/secure_grpc_pb2*.py
python -c "from src.omnifed.communicator.secure_grpc import secure_grpc_pb2_grpc as g; print(g.SecureGrpcServerStub)"
```

Notes:

* `-I.` plus the repo-relative path matters. It makes the proto's registered name
  `src/omnifed/communicator/secure_grpc/secure_grpc.proto`, and the generated `_pb2_grpc.py` imports
  `from src.omnifed.communicator.secure_grpc import secure_grpc_pb2`, which works because the repo
  root is on `sys.path` (the existing `grpc_pb2_grpc.py` relies on the same thing).
* Package, service and file name are all different from `grpc.proto`, so both can be loaded in one
  process (no "duplicate symbol" error, no clash on RPC paths).
* `main.sh` regenerates only `grpc.proto` and `global_grpc.proto`, and I did not edit it. If you want
  the secure code regenerated on every launch (e.g. on Frontier), add one line next to the others:

  ```bash
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./src/omnifed/communicator/secure_grpc/secure_grpc.proto
  ```

* `grpc_pb2.py` is tracked in git in this repo; whether to track the new generated files is your call.
* Do not hand-edit the generated files; change the `.proto` and regenerate.
* `TensorEntry` / `TensorDict` in `secure_grpc.proto` are copies of the ones in `grpc.proto`, because the
  shared decoders in `communicator/utils.py` read them by field name. If you change one proto, change
  the other; `SecureGrpcCommunicator.setup()` compares them and refuses to start if they differ.

| If you see | Cause |
|---|---|
| `No module named grpc_tools` | `pip install grpcio-tools==1.81.0` in that environment |
| `ImportError: ... secure_grpc_pb2` when starting | the command above has not been run (or ran in another directory) |
| `Protobuf gencode version ... runtime version` error | generated with different tools than the ones you run with; regenerate in the run environment |

## 2. Certificates for the communicator

Naming rule: rank N must present a certificate whose CN is `client-N` (rank 1 -> `client-1`, ...).
The server's certificate has no CN requirement but its SAN must contain the host the clients dial.

```bash
bash src/omnifed/communicator/secure_grpc/gen_certs.sh \
    -o src/omnifed/communicator/secure_grpc/secure_grpc_certs \
    -s <host the clients will dial, if not this machine's hostname> \
    client-1 client-2
```

`.gitignore` already ignores `src/omnifed/communicator/secure_grpc/secure_grpc_certs*/`. For a real
deployment put each node's files in the same layout and copy nothing else:

| Node | Needs |
|---|---|
| server (rank 0) | `ca.crt`, `server/server.crt`, `server/server.key` |
| client N | `ca.crt`, `clients/client-N.crt`, `clients/client-N.key` |

(See "What each certificate and key file is" above for what these are and who may see them. Keys must
be mode 600 and unencrypted; the communicator refuses a group/world-readable key.)

## 3. Smoke test: server + 2 clients

After step 1 and 2, three terminals (same machine, or three login nodes with the files above in place):

```bash
export CERTS=$PWD/src/omnifed/communicator/secure_grpc/secure_grpc_certs
S=src/omnifed/communicator/secure_grpc/smoke_secure_grpc.py

python $S --rank 0 --world-size 3 --tls-dir $CERTS                       # terminal 1
python $S --rank 1 --world-size 3 --tls-dir $CERTS                       # terminal 2
python $S --rank 2 --world-size 3 --tls-dir $CERTS                       # terminal 3
```

For separate nodes add `--addr <server host>` to every command (that name must be in the server
certificate's SAN). Each rank prints `broadcast OK`, `round N: mean ... OK` and `PASS`. Add
`--force-chunking` to push the same data through the streaming RPCs. To watch a wrong certificate
being refused, start rank 0 and then run rank 2 with
`--tls-cert .../clients/client-1.crt --tls-key .../clients/client-1.key`.

## 4. Using it in OmniFed

Copy `secure_grpc.example.yaml` to `conf/communicator/secure_grpc.yaml` and attach it where you attach
`grpc.yaml` today (`- /communicator@topology.local_comm: secure_grpc`), or write the `_target_` inline:

```yaml
topology:
  local_comm:
    _target_: src.omnifed.communicator.secure_grpc.secure_grpc_communicator.SecureGrpcCommunicator
    master_addr: <host in the server certificate SAN>
    master_port: 50443
    tls_dir: ${oc.env:OMNIFED_TLS_DIR}
```

`tls_dir` makes one shared config work on every rank (each rank reads only its own files). Everything
else can be given explicitly (`tls_ca_cert`, `tls_cert`, `tls_key`, which may contain `{rank}`), or through
the environment variables `OMNIFED_TLS_CA` / `OMNIFED_TLS_CERT` / `OMNIFED_TLS_KEY`. Precedence:
explicit argument, then `tls_dir`, then environment. All constructor arguments of `GrpcCommunicator`
are accepted unchanged; the new ones are:

| Argument | Default | Meaning |
|---|---|---|
| `tls_dir`, `tls_ca_cert`, `tls_cert`, `tls_key` | none | where this rank's certificate material is (see above) |
| `client_cn_prefix` | `client-` | certificate CN `<prefix><rank>` is rank `<rank>` |
| `client_identities` | none | `{CN: rank}` mapping instead of the prefix rule (each rank at most once) |
| `rpc_timeout` | 5000 s | deadline on every data-carrying RPC |
| `connect_timeout` | 30 s | TLS connect and `RegisterClient` |
| `long_poll_sec` | 30 s | how long the server holds one "result ready?" request before answering "not yet" |
| `require_client_deadline` | true | server refuses calls that carry no deadline |
| `max_concurrent_rpcs` | `max(max_workers, clients + 4)` | in-flight RPC cap on the server; excess calls get `RESOURCE_EXHAUSTED` |
| `max_concurrent_streams` | 16 | HTTP/2 streams per connection |
| `keepalive_sec` | 60 | server keepalive pings (reaps dead peers); `None` disables |
| `handshake_timeout_sec` | 20 | a TCP peer that never completes TLS is dropped after this long |
| `idle_timeout_sec` | none | close connections idle this long (off: training can be quiet for hours) |
| `bind_addr` | `[::]` | interface the server listens on |
| `tls_server_name_override` | none | **testing only**: name to verify the server certificate against |
| `tls_allow_loose_key_permissions` | false | accept a group/world-readable private key (not recommended) |

## 5. What is different from the existing `GrpcCommunicator`

Aggregation, sessions, sample weighting, compression, chunking and the public API are the existing code;
the fork was built by slicing your methods and patching them, so the unchanged logic is identical.

| Area | `GrpcCommunicator` | `SecureGrpcCommunicator` |
|---|---|---|
| Transport | `add_insecure_port` / `insecure_channel` | `add_secure_port` / `secure_channel`, mTLS only (TLS 1.2+), no plaintext port |
| Server authenticity | none | client checks CA chain and that the dialled host is in the server certificate SAN |
| Client authentication | none | `require_client_auth=True`: a certificate signed by the CA is needed to finish the handshake |
| Caller identity | `client_id` field in the request (`str(rank)`), self-asserted | no id field in any request; identity is the CN of the verified certificate from `auth_context()` |
| Authorization | `registered_clients` is only counted | CN must be in the allowlist (`client-<rank>` by default); checked first in every RPC |
| Wrong rank / certificate mix-up | undetected | caught before connecting (CN pre-check, needs `cryptography`) and again by the server's `RegisterClient` reply |
| Waiting for the result | `event.wait()` with no timeout holds a worker thread until the round ends | waits at most `long_poll_sec` and never past the client's deadline; wakes if the client hangs up; the client asks again |
| Deadlines | none on any RPC | `rpc_timeout` on every data RPC, `connect_timeout` for connecting; calls without a deadline are refused |
| Concurrency | 10 workers, no cap | `maximum_concurrent_rpcs`, pool never smaller than the cap, refuses a cap below the clients that wait at once |
| Failed authentication | n/a | `UNAUTHENTICATED` / `PERMISSION_DENIED` are not retried and say what to check; "server busy" is retried |
| Port | `[::]`, silently shares a port with another server | `grpc.so_reuseport=0`: a second server on the port fails loudly |
| Startup checks | none | key permissions, passphrase-protected key, PEM format; with `cryptography`: key matches certificate, not expired / not yet valid, CN is right for this rank |
| Logging | prints the caller-supplied id | `DENY` lines with peer address for refused identities (standard `logging`, logger `omnifed.secure_grpc`) |

## 6. Scope and limits

* An authenticated client is trusted to send well-formed tensors. A certified but malicious or buggy
  client can still send mismatched shapes (which abandons the round) or a Top-K `original_shape` that
  makes the server allocate a lot; the message-size cap bounds the damage but does not remove it.
* No revocation list: to cut off a client, remove its name from the allowlist (or reissue the CA).
  Certificates are read once at start-up; rotating them needs a restart. Identity is the CN, so use a
  CA that issues only OmniFed certificates.
* After a long poll expires the client sleeps `retry_delay` before asking again, so with the default
  5 s a result that becomes ready in that gap is seen up to 5 s late (only when a round takes longer
  than `long_poll_sec`).

## 7. How this part was tested, and what was not

Run in a sandbox without torch, `protoc` or the repo's other dependencies:

* `test_security.py` (shipped): real gRPC servers over real mTLS. Identity comes from the certificate,
  an unlisted but valid certificate is refused, a foreign CA / no client certificate / plaintext are
  refused, hostname verification is active, deadlines are enforced, the concurrent-RPC cap rejects the
  excess, a stalled handshake is dropped, a second server cannot share the port.
* The real `secure_grpc_{communicator,server,client}.py`, `secure_codec.py` and `security.py` were run
  against stand-ins for torch and for the generated protobuf code (message classes built from this
  `.proto`). Passed: server + 2 clients end to end (broadcast, two MEAN rounds, SUM), the same through the
  streaming RPCs, wrong-rank and unlisted certificates, foreign CA, a client 3.5 s late with
  `long_poll_sec=1`, calls without a deadline, authentication on all five RPC shapes, bad configuration
  failing at start-up, 127.0.0.1 without a name override, and `smoke_secure_grpc.py` as three processes.
* **Not run:** real torch tensors, the protoc-generated modules, Top-K / QSGD compression, GPU
  `agg_device`, Slurm / Frontier networking, and a real model. `smoke_secure_grpc.py` is how you
  confirm the real thing in your environment; if something differs, the message will say which step.

## 8. Troubleshooting

| Message | Meaning |
|---|---|
| `Server refused this client (PERMISSION_DENIED: identity not allowed)` | the certificate's CN is not an allowed name: check `client-<rank>` naming, `world_size`, `client_identities` |
| `The server registered this client as participant '1' but this is rank '2'` | rank 2 is using rank 1's certificate files |
| `certificate ... has CN='client-1' but rank 2 must present CN='client-2'` | same, caught before connecting |
| `Failed to connect ... TLS/connection failure. Check: ...` | server down/unreachable, host not in the server SAN, CA mismatch, expired certificate or clock skew |
| `private key ... accessible by group/others` | `chmod 600` the key |
| `Cannot listen on [::]:50443 (port in use ...)` | another process (or server) already owns the port |
| `RPC deadline required` | something called the server without a timeout (other tooling); set one, or `require_client_deadline: false` |
| `max_concurrent_rpcs=N is below the K clients` | raise `max_concurrent_rpcs` / `max_workers` |
| `secure_grpc.proto TensorEntry differs from grpc.proto` | the two protos drifted; make the tensor messages identical and regenerate |