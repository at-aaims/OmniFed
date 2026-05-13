import os
import socket

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "29500")
    hostname = socket.gethostname()

    print(
        f"[rank {rank}/{world}] host={hostname} local_rank={local_rank} "
        f"master={master_addr}:{master_port} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}",
        flush=True,
    )

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    print(f"[rank {rank}] process group initialized (nccl)", flush=True)

    x = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world + 1))
    print(
        f"[rank {rank}] all_reduce result={x.item()} (expected {expected})",
        flush=True,
    )

    dist.destroy_process_group()
    print(f"[rank {rank}] done.", flush=True)


if __name__ == "__main__":
    main()
