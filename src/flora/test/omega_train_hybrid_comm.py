import logging
import socket
from time import perf_counter_ns
from typing import Any, Dict, Optional, Tuple

import torch

from src.flora.test import get_model
from src.flora.communicator import torch_mpi
from src.flora.communicator import grpc_communicator as rpc_comm
from src.flora.datasets.image_classification import cifar

nanosec_to_millisec = 1e6


def _find_facility(cfg, global_rank: int) -> Optional[Dict[str, Any]]:
    for fac in cfg.topology.facilities:
        if global_rank in list(fac.mpi.members):
            return fac
    return None

def _derive_local_rank(facility, global_rank: int) -> int:
    members = list(facility.mpi.members)
    if global_rank not in members:
        raise RuntimeError(f"global_rank={global_rank} not in facility members={members}")
    return members.index(global_rank)  # leader should be index 0 if config is consistent


class HybridTrainer(object):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.global_rank = args.global_rank
        self.local_rank = args.local_rank
        self.backend = args.backend
        self.epochs = args.epochs
        self.comm_freq = args.comm_freq

        # CPU-only (as you requested)
        self.device = torch.device("cpu")

        logging.basicConfig(
            filename=self.logdir
            + "/g"
            + str(self.global_rank)
            + "/"
            + self.model_name
            + "-"
            + str(self.global_rank)
            + ".log",
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
        # Topology-driven communicator setup
        # -----------------------------
        topo = cfg.topology

        rpc_server_rank = int(topo.rpc.server_rank)
        rpc_addr = str(topo.rpc.addr)
        rpc_port = int(topo.rpc.port)
        rpc_client_ranks = set(int(r) for r in topo.rpc.client_ranks)

        facility = _find_facility(cfg, self.global_rank)
        in_mpi = facility is not None
        is_rpc_server = self.global_rank == rpc_server_rank
        is_rpc_client = self.global_rank in rpc_client_ranks

        self.mpi_comm = None
        self.rpc_comm = None

        # gRPC: server + clients count
        rpc_total_processes = 1 + len(rpc_client_ranks)

        if is_rpc_server:
            # Server only (matches your current behavior)
            self.mpi_comm = None
            self.rpc_comm = rpc_comm.GrpcCommunicator(
                model=self.model,
                id=self.global_rank,
                total_clients=rpc_total_processes,
                master_addr=rpc_addr,
                master_port=rpc_port,
                accumulate_updates=True,
            )
            logging.info(f"started gRPC server on global_rank {self.global_rank}...")

        if in_mpi:
            # MPI communicator for this facility
            mpi_addr = str(facility.mpi.addr)
            mpi_port = int(facility.mpi.port)
            mpi_world_size = int(facility.mpi.world_size)
            self.local_rank = _derive_local_rank(facility, self.global_rank)

            self.mpi_comm = torch_mpi.TorchMPICommunicator(
                id=self.local_rank,
                total_clients=mpi_world_size,
                backend=self.backend,
                master_addr=mpi_addr,
                master_port=mpi_port,
            )
            logging.info(
                f"started MPI process on global_rank {self.global_rank} "
                f"(facility={facility.name}, local_rank={self.local_rank}, world_size={mpi_world_size})..."
            )

            # gRPC client (facility leaders in your topology)
            if is_rpc_client:
                self.rpc_comm = rpc_comm.GrpcCommunicator(
                    model=self.model,
                    id=self.global_rank,
                    total_clients=rpc_total_processes,
                    master_addr=rpc_addr,
                    master_port=rpc_port,
                    accumulate_updates=True,
                )
                logging.info(f"started gRPC client on global_rank {self.global_rank}...")

        logging.info("initialized communicator object...")

        # -----------------------------
        # Dataloader (kept as-is)
        # -----------------------------
        num_trainers = cfg.training.dataset_total_clients  # 6 in symmetric
        if self.global_rank == cfg.topology.rpc.server_rank:
            # server shouldn't be a data trainer; give it no loader OR a harmless one
            self.train_dataloader, self.test_dataloader = None, None
        else:
            trainer_id = self.global_rank - 1
            self.train_dataloader, self.test_dataloader = cifar.cifar10Data(
                    client_id=trainer_id,
                    total_clients=int(cfg.training.dataset_total_clients),
                    datadir=args.dir,
                    train_bsz=self.train_bsz,
                    test_bsz=self.test_bsz,
                    partition_dataset=True,
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
                if self.mpi_comm is not None:
                    init_time = perf_counter_ns()
                    self.model = self.mpi_comm.aggregate(
                        msg=self.model, communicate_params=True, compute_mean=True
                    )
                    mpi_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec
                if self.mpi_comm is not None and (self.local_step % self.comm_freq == 1):
                    with torch.no_grad():
                        w = dict(self.model.named_parameters())["conv1.weight"]
                        norm = torch.norm(w).item()
                        print(f"[Rank {self.global_rank}] AFTER MPI all_reduce(mean) step={self.local_step} norm={norm:.6f}", flush=True)
                    #mpi_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                rpc_sync_time = None
                if self.local_step % self.comm_freq == 0:
                    init_time = perf_counter_ns()
                    print("going to initial rpc call now...")
                    facility_samples = self.training_samples
                    
                    if self.mpi_comm is not None:
                        sample_t = torch.tensor([self.training_samples], dtype=torch.long)
                        torch.distributed.all_reduce(sample_t, op=torch.distributed.ReduceOp.SUM)
                        facility_samples = int(sample_t.item())


                    # Only gRPC server + gRPC clients participate
                    if self.rpc_comm is not None:
                        self.model = self.rpc_comm.aggregate(
                            msg=self.model,
                            communicate_params=True,
                            #compute_mean=False,
                            batch_samples=facility_samples,
                        )
                        #self.training_samples = 0
                        rpc_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                    # MPI broadcast within each facility group:
                    # root is local_rank==0 i.e. src=0 in the MPI group.
                    # ALL ranks in that MPI group must participate.
                    if self.mpi_comm is not None:
                        self.model = self.mpi_comm.broadcast(msg=self.model, id=0)
                        # ---- MPI sanity check (inside the MPI block) ----
                        with torch.no_grad():
                            param = dict(self.model.named_parameters())["conv1.weight"]
                            norm = torch.norm(param).item()
                            print(f"[Rank {self.global_rank}] conv1.weight norm after sync: {norm:.6f}")

                    # ---- Reset training sample counter for next gRPC window ----
                    if self.mpi_comm is not None:
                        self.training_samples = 0
                    elif self.rpc_comm is not None:
                        # gRPC-only client (unlikely in your topology)
                        self.training_samples = 0


                logging.info(
                    f"training_metrics local_step: {self.local_step} compute_time {compute_time} ms "
                    f"mpi_sync_time: {mpi_sync_time} ms rpc_sync_time: {rpc_sync_time} ms"
                )

