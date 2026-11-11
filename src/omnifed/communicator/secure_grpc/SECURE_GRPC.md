# Secure gRPC (mTLS) standalone test

A self-contained prototype of what OmniFed's gRPC aggregation could look like with
mutual TLS, caller identity from the certificate, explicit deadlines and bounded
concurrency. It does **not** import OmniFed, protobuf or torch (only `grpcio` + `numpy`),
and it does not touch the existing communicators.

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