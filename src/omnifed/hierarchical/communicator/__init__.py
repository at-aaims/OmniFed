# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Hybrid Slurm facility-local TorchMPI and global gRPC (FedAvg across facilities)."""


class Communicator(object):
    def __init__(self, protocol_type):
        self.protocol_type = protocol_type

    def broadcast(self, **kwargs):
        raise NotImplementedError("implement broadcast")

    def send(self, **kwargs):
        raise NotImplementedError("implement send")

    def receive(self, **kwargs):
        raise NotImplementedError("implement receive")

    def aggregate(self, **kwargs):
        raise NotImplementedError("implement aggregate")

    def allgather(self, **kwargs):
        raise NotImplementedError("implement allgather")

    def close(self, **kwargs):
        raise NotImplementedError("implement close")
