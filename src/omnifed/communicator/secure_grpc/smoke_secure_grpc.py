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

"""Smoke test of SecureGrpcCommunicator with real torch and the protoc-generated code.

Run AFTER generating secure_grpc_pb2*.py (see SECURE_GRPC.md). One process per rank, exactly
like a real run. Three terminals (or three login nodes), same --tls-dir layout on each:

    python smoke_secure_grpc.py --rank 0 --world-size 3 --tls-dir CERTS --addr SERVER_HOST
    python smoke_secure_grpc.py --rank 1 --world-size 3 --tls-dir CERTS --addr SERVER_HOST
    python smoke_secure_grpc.py --rank 2 --world-size 3 --tls-dir CERTS --addr SERVER_HOST

Every rank broadcasts / aggregates a tensor, checks the numbers, and exits 0 on success.
--force-chunking pushes the same data through the streaming RPCs. Overrides such as
--tls-cert / --tls-key let you try a wrong certificate and watch it being refused.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# <repo>/src/omnifed/communicator/secure_grpc/this_file -> repo root is parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch  # noqa: E402

from src.omnifed.communicator.base import AggregationOp  # noqa: E402
from src.omnifed.communicator.secure_grpc.secure_grpc_communicator import (  # noqa: E402
    SecureGrpcCommunicator,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rank", type=int, required=True, help="0 = server")
    p.add_argument("--world-size", type=int, required=True, help="server + clients")
    p.add_argument("--addr", default="localhost", help="server host; must be in its certificate SAN [localhost]")
    p.add_argument("--port", type=int, default=50443)
    p.add_argument("--tls-dir", help="gen_certs.sh layout: ca.crt, server/server.*, clients/client-N.*")
    p.add_argument("--tls-ca", help="override CA file")
    p.add_argument("--tls-cert", help="override this rank's certificate")
    p.add_argument("--tls-key", help="override this rank's private key")
    p.add_argument("--server-name-override", help="TESTING ONLY: name to verify the server certificate against")
    p.add_argument("--elements", type=int, default=1000, help="tensor length [1000]")
    p.add_argument("--rounds", type=int, default=2, help="aggregation rounds [2]")
    p.add_argument("--rpc-timeout", type=float, default=120.0, help="deadline on every data RPC, s [120]")
    p.add_argument("--long-poll-sec", type=float, default=30.0, help="server-side wait per result request, s [30]")
    p.add_argument("--force-chunking", action="store_true", help="tiny chunk thresholds: exercise the streaming RPCs")
    p.add_argument("--linger", type=float, default=5.0, help="server waits this long before closing [5]")
    args = p.parse_args(argv)

    extra = {}
    if args.force_chunking:
        extra = dict(chunk_threshold_bytes=256, chunk_payload_bytes=128)

    comm = SecureGrpcCommunicator(
        rank=args.rank,
        world_size=args.world_size,
        master_addr=args.addr,
        master_port=args.port,
        tls_dir=args.tls_dir,
        tls_ca_cert=args.tls_ca,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        tls_server_name_override=args.server_name_override,
        rpc_timeout=args.rpc_timeout,
        long_poll_sec=args.long_poll_sec,
        retry_delay=1.0,
        max_retries=30,
        **extra,
    )
    ok = True
    try:
        comm.setup()  # server: opens the secure port. client: mTLS connect + register.

        n = args.elements
        reference = torch.arange(n, dtype=torch.float32)
        got = comm.broadcast({"w": reference.clone() if args.rank == 0 else torch.zeros(n)})
        good = torch.allclose(got["w"], reference)
        print(f"[rank {args.rank}] broadcast {'OK' if good else 'WRONG'}")
        ok &= good

        for r in range(1, args.rounds + 1):
            local = {"w": torch.full((n,), float((args.rank + 1) * r))}
            agg = comm.aggregate(local, AggregationOp.MEAN)
            expected = r * (args.world_size + 1) / 2  # mean over ranks of (rank+1)*r
            good = torch.allclose(agg["w"], torch.full((n,), expected))
            print(f"[rank {args.rank}] round {r}: mean {agg['w'][0].item():.4f} (expected {expected:.4f}) {'OK' if good else 'WRONG'}")
            ok &= good
        if args.rank == 0:
            time.sleep(args.linger)  # let clients finish fetching before the server closes
    finally:
        comm.close()

    print(f"[rank {args.rank}] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
