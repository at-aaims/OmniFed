import logging
import socket
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Dict, List, Optional, Tuple

import torch

from src.flora.test import get_model
from src.flora.communicator import torch_mpi
from src.flora.communicator import grpc_communicator as rpc_comm
from src.flora.datasets.image_classification import cifar

nanosec_to_millisec = 1e6


# -----------------------------
# Topology config (default: your current 7-process setup)
# -----------------------------
@dataclass(frozen=True)
class HybridTopo:
    # Global ranks
    rpc_server_rank: int
    rpc_client_ranks: Tuple[int, ...]  # ranks that talk to gRPC server
    mpi_groups: Dict[str, Tuple[int, ...]]  # group_name -> global ranks in that MPI group
    mpi_leaders: Dict[str, int]  # group_name -> global rank that is leader (local_rank==0)
    # Networking
    rpc_master_addr: str
    rpc_master_port: int
    mpi_master_addrs: Dict[str, str]  # group_name -> addr
    mpi_master_ports: Dict[str, int]  # group_name -> port
    # Dataset/training clients (not including rpc server-only process)
    train_worker_ranks: Tuple[int, ...]  # ranks that actually train
    dataset_total_clients: int  # used in cifar.cifar10Data(total_clients=...)
    # MPI group size (all groups same size in your current design)
    mpi_world_size: int


def default_topology() -> HybridTopo:
    """
    Keeps your existing behavior exactly:
      - global ranks: 0..6
      - rank0 = gRPC server only
      - ranks1,2 = MPI leaders + gRPC clients
      - ranks3,4 = MPI group1 workers
      - ranks5,6 = MPI group2 workers
      - each MPI group has world_size=3 with local ranks {0,1,2}
      - CIFAR total_clients=6 (workers are ranks 1..6)
    """
    mpi_group1 = (1, 3, 4)  # leader first (global 1), then workers
    mpi_group2 = (2, 5, 6)  # leader first (global 2), then workers

    return HybridTopo(
        rpc_server_rank=0,
        rpc_client_ranks=(1, 2),
        mpi_groups={"g1": mpi_group1, "g2": mpi_group2},
        mpi_leaders={"g1": 1, "g2": 2},
        rpc_master_addr="127.0.0.1",
        rpc_master_port=50051,
        mpi_master_addrs={"g1": "127.0.0.1", "g2": "127.0.0.1"},
        mpi_master_ports={"g1": 28250, "g2": 28290},
        train_worker_ranks=(1, 2, 3, 4, 5, 6),
        dataset_total_clients=6,
        mpi_world_size=3,
    )


def find_mpi_group(topo: HybridTopo, global_rank: int) -> Optional[str]:
    for name, members in topo.mpi_groups.items():
        if global_rank in members:
            return name
    return None


def is_mpi_leader(topo: HybridTopo, group: str, global_rank: int) -> bool:
    return topo.mpi_leaders[group] == global_rank


def compute_local_rank_from_group(topo: HybridTopo, group: str, global_rank: int) -> int:
    """
    Derive local_rank from topology, instead of passing it in from bash.
    This removes hardcoded localranks array from the launcher logic (later).
    For now, it also acts as a consistency check against args.local_rank.
    """
    members = topo.mpi_groups[group]
    return members.index(global_rank)  # leader is 0 by convention


def pick_device(global_rank: int) -> torch.device:
    """
    Avoid 'global_rank % 7' which is brittle.
    Default mapping:
      - if CUDA available: use cuda:0 for all ranks unless LOCAL_RANK is set
        (you can refine later when you switch to torchrun/mpirun env vars).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Prefer LOCAL_RANK if set (torchrun / some launchers)
    try:
        import os
        lr = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        lr = -1

    if lr >= 0:
        return torch.device(f"cuda:{lr}")

    # Fallback: map global rank to available device count (safe even if !=7 GPUs)
    n = torch.cuda.device_count()
    if n <= 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{global_rank % n}")


class HybridTrainer(object):
    def __init__(self, args):
        self.args = args
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.global_rank = args.global_rank
        self.local_rank = args.local_rank  # will be checked vs derived when in MPI group
        self.backend = args.backend
        self.epochs = args.epochs
        self.comm_freq = args.comm_freq

        # Single place to define your topology (still 7-process by default)
        self.topo = default_topology()

        # Device selection (safer than %7)
        self.device = pick_device(self.global_rank)

        logging.basicConfig(
            filename=f"{self.logdir}/g{self.global_rank}/{self.model_name}-{self.global_rank}.log",
            level=logging.INFO,
        )

        self.dataset_name = args.dataset
        self.determinism = False

        self.model_obj = get_model(self.model_name, determinism=self.determinism, args=args)
        if self.model_obj is None:
            raise RuntimeError(f"invalid model or model_name {self.model_name}")

        logging.info("initialized model object...")
        self.model = self.model_obj.get_model().to(self.device)
        self.loss_fn = self.model_obj.get_loss()
        self.optimizer = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.local_step = 0
        self.training_samples = 0

        # -----------------------------
        # Role selection based on topology (no hardcoded rank branches)
        # -----------------------------
        self.mpi_comm = None
        self.rpc_comm = None

        group = find_mpi_group(self.topo, self.global_rank)

        # 1) gRPC server role
        if self.global_rank == self.topo.rpc_server_rank:
            self.rpc_comm = rpc_comm.GrpcCommunicator(
                model=self.model,
                id=self.global_rank,
                total_clients=1 + len(self.topo.rpc_client_ranks),  # server + clients
                master_addr=self.topo.rpc_master_addr,
                master_port=self.topo.rpc_master_port,
                accumulate_updates=True,
            )
            logging.info("started gRPC server...")

        # 2) MPI member role (leaders/workers)
        if group is not None:
            derived_local_rank = compute_local_rank_from_group(self.topo, group, self.global_rank)

            # Optional safety check: your bash currently passes local_rank explicitly.
            # This helps catch mismatches.
            if self.local_rank != derived_local_rank:
                logging.warning(
                    f"local_rank mismatch: args.local_rank={self.local_rank} "
                    f"but derived_local_rank={derived_local_rank} from topo group={group}. "
                    f"(Proceeding with derived_local_rank.)"
                )

            self.local_rank = derived_local_rank

            self.mpi_comm = torch_mpi.TorchMPICommunicator(
                id=self.local_rank,
                total_clients=self.topo.mpi_world_size,
                backend=self.backend,
                master_addr=self.topo.mpi_master_addrs[group],
                master_port=self.topo.mpi_master_ports[group],
            )
            logging.info(f"started MPI process in group={group} local_rank={self.local_rank}...")

            # MPI leaders are also gRPC clients
            if self.global_rank in self.topo.rpc_client_ranks:
                self.rpc_comm = rpc_comm.GrpcCommunicator(
                    model=self.model,
                    id=self.global_rank,
                    total_clients=1 + len(self.topo.rpc_client_ranks),  # server + clients
                    master_addr=self.topo.rpc_master_addr,
                    master_port=self.topo.rpc_master_port,
                    accumulate_updates=True,
                )
                logging.info("started gRPC client...")

        logging.info("initialized communicator object...")

        # -----------------------------
        # Dataloader (kept as-is)
        # -----------------------------
        self.train_dataloader, self.test_dataloader = cifar.cifar10Data(
            client_id=self.global_rank,
            total_clients=self.topo.dataset_total_clients,
            datadir=args.dir,
            train_bsz=self.train_bsz,
            test_bsz=self.test_bsz,
            partition_dataset=False,
        )
        logging.info("initialized dataloader object...")

        args.hostname = socket.gethostname()
        args.optimizer = self.optimizer.__class__.__name__
        logging.info(f"training/job specific parameters: {args}")

    def start_training(self):
        for epoch in range(self.epochs):
            for inputs, labels in self.train_dataloader:
                init_time = perf_counter_ns()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                pred = self.model(inputs)
                loss = self.loss_fn(pred, labels)

                self.training_samples += inputs.size(0)
                loss.backward()

                compute_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.local_step += 1

                mpi_sync_time = None
                # Only ranks that are in an MPI group do MPI all-reduce
                if self.mpi_comm is not None:
                    init_time = perf_counter_ns()
                    self.model = self.mpi_comm.aggregate(
                        msg=self.model, communicate_params=True, compute_mean=False
                    )
                    mpi_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                rpc_sync_time = None
                if self.local_step % self.comm_freq == 0:
                    init_time = perf_counter_ns()
                    print("going to initial rpc call now...")

                    # Only gRPC server + gRPC clients participate
                    if self.rpc_comm is not None:
                        self.model = self.rpc_comm.aggregate(
                            msg=self.model,
                            communicate_params=True,
                            compute_mean=False,
                            batch_samples=self.training_samples,
                        )
                        self.training_samples = 0
                        rpc_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                    # MPI leaders broadcast the (potentially) gRPC-updated model to their local MPI group
                    if self.mpi_comm is not None and self.local_rank == 0:
                        self.model = self.mpi_comm.broadcast(msg=self.model, id=self.local_rank)

                logging.info(
                    f"training_metrics local_step: {self.local_step} compute_time {compute_time} ms "
                    f"mpi_sync_time: {mpi_sync_time} ms rpc_sync_time: {rpc_sync_time} ms"
                )

