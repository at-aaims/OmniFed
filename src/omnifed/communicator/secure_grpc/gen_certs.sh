#!/usr/bin/env bash
# gen_certs.sh -- throw-away CA + server certificate + one certificate per client,
# for the mutual-TLS (mTLS) secure gRPC test.  Needs only bash and openssl >= 1.1.1.
#
# Usage:
#   bash gen_certs.sh [-o OUTDIR] [-s EXTRA_SAN] [-d DAYS] CLIENT_NAME [CLIENT_NAME ...]
#
#   -o OUTDIR    output directory (default: ./certs); refuses to overwrite an existing CA
#   -s EXTRA_SAN extra Subject Alternative Names for the SERVER cert, comma separated,
#                e.g. "DNS:login05.frontier.olcf.ornl.gov,IP:10.1.2.3".  The name the
#                clients use in --server HOST:PORT MUST appear here (or in the defaults).
#   -d DAYS      validity of every certificate (default: 30; short-lived on purpose,
#                gRPC-Python has no CRL/OCSP support, so expiry is your revocation)
#
# Defaults for the server SAN are taken from this machine: hostname -f, hostname -s,
# localhost, 127.0.0.1.  Run this on the node that will host the server, or pass -s.
#
# Layout produced:
#   OUTDIR/ca.crt  ca.key                  CA (keep ca.key private / off the nodes)
#   OUTDIR/server/server.crt  server.key   server identity   (EKU serverAuth)
#   OUTDIR/clients/<name>.crt <name>.key   one identity per client (EKU clientAuth);
#                                          the certificate CN is the identity the server sees
#
# TEST ONLY: this script creates every private key in one place.  In a real multi-site
# deployment each facility generates its own key + CSR and sends ONLY the CSR to the CA;
# a private key never leaves its owner.

set -euo pipefail
umask 077

OUT="certs"
DAYS=30
EXTRA_SAN=""

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

while getopts "o:s:d:h" opt; do
  case "$opt" in
    o) OUT=$OPTARG ;;
    s) EXTRA_SAN=$OPTARG ;;
    d) DAYS=$OPTARG ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || usage 1

command -v openssl >/dev/null || { echo "openssl not found in PATH" >&2; exit 1; }
[[ "$DAYS" =~ ^[0-9]+$ ]] || { echo "-d must be an integer" >&2; exit 1; }

for name in "$@"; do
  [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad client name '$name' (use A-Z a-z 0-9 _ . -)" >&2; exit 1; }
done
[ ! -e "$OUT/ca.crt" ] || { echo "refusing to overwrite existing CA in '$OUT' (remove it or pick -o)" >&2; exit 1; }

# run a command quietly, but show its output if it fails
quiet() { local out; if ! out=$("$@" 2>&1); then echo "$out" >&2; return 1; fi; }

HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
SAN="DNS:${HOST_FQDN},DNS:${HOST_SHORT},DNS:localhost,IP:127.0.0.1"
[ -z "$EXTRA_SAN" ] || SAN="${SAN},${EXTRA_SAN}"
SAN="$(printf '%s' "$SAN" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)"

mkdir -p "$OUT/server" "$OUT/clients"

newkey() { quiet openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$1"; }

# --- CA ---------------------------------------------------------------------------
newkey "$OUT/ca.key"
quiet openssl req -x509 -new -key "$OUT/ca.key" -sha256 -days "$DAYS" \
  -subj "/CN=OmniFed-secure-grpc-test-CA" \
  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -out "$OUT/ca.crt"

# issue KEYFILE CRTFILE COMMON_NAME EXTENSIONS(newline separated)
issue() {
  local key=$1 crt=$2 cn=$3 ext=$4 csr extf
  csr="$(mktemp)"; extf="$(mktemp)"
  printf '%s\n' "$ext" > "$extf"
  newkey "$key"
  quiet openssl req -new -key "$key" -subj "/CN=${cn}" -out "$csr"
  quiet openssl x509 -req -in "$csr" -CA "$OUT/ca.crt" -CAkey "$OUT/ca.key" \
    -set_serial "0x$(openssl rand -hex 16)" -days "$DAYS" -sha256 -extfile "$extf" -out "$crt"
  rm -f "$csr" "$extf"
}

# --- server -----------------------------------------------------------------------
issue "$OUT/server/server.key" "$OUT/server/server.crt" "omnifed-server" \
$'basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\nsubjectAltName='"$SAN"

# --- clients ----------------------------------------------------------------------
for name in "$@"; do
  issue "$OUT/clients/$name.key" "$OUT/clients/$name.crt" "$name" \
$'basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=clientAuth\nsubjectAltName=DNS:'"$name"
done

chmod 600 "$OUT"/ca.key "$OUT"/server/server.key "$OUT"/clients/*.key
chmod 644 "$OUT"/ca.crt "$OUT"/server/server.crt "$OUT"/clients/*.crt

# --- self-check -------------------------------------------------------------------
quiet openssl verify -CAfile "$OUT/ca.crt" -purpose sslserver "$OUT/server/server.crt"
for name in "$@"; do
  quiet openssl verify -CAfile "$OUT/ca.crt" -purpose sslclient "$OUT/clients/$name.crt"
done

fp() { openssl x509 -in "$1" -noout -fingerprint -sha256 | sed 's/.*=//'; }
echo "Certificates written to: $OUT   (valid ${DAYS} days)"
echo "  CA      $OUT/ca.crt                      sha256 $(fp "$OUT/ca.crt")"
echo "  server  $OUT/server/server.crt           SAN: $SAN"
for name in "$@"; do
  echo "  client  $OUT/clients/$name.crt   CN=$name   sha256 $(fp "$OUT/clients/$name.crt")"
done
echo
echo "Server:  python test_grpc_server.py --ca $OUT/ca.crt --cert $OUT/server/server.crt --key $OUT/server/server.key \\"
echo "             --allowed-clients $(IFS=,; echo "$*")"
echo "Client:  python test_grpc_client.py --server <HOST-from-SAN>:50051 --ca $OUT/ca.crt \\"
echo "             --cert $OUT/clients/<name>.crt --key $OUT/clients/<name>.key --index <i> --num-clients $#"
