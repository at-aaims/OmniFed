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

import copy
import hmac
import random
import logging
import hashlib
from typing import Dict, List
from time import perf_counter_ns

import torch

from src.flora.helper import compute_gradient_norm
from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.flora.helper.training_params import FedAvgTrainingParameters
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)


nanosec_to_millisec = 1e6


class SecureAggregation:
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        communicator: TorchMPICommunicator,
        total_clients: int,
        train_params: FedAvgTrainingParameters,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.lr_scheduler = self.train_params.get_lr_scheduler()
        self.epochs = self.train_params.get_epochs()
        self.local_step = 0
        self.training_samples = 0
        self.client_id = client_id
        dev_id = self.client_id % 4
        self.device = (
            torch.device("cuda:" + str(dev_id))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.top1_acc, self.top5_acc, self.top10_acc = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )

        # number of clients needed to reconstruct updates
        self.threshold_t = self.total_clients
        # Use a large prime for finite field arithmetic (2^31 - 1 for simplicity)
        self.modulus_q = 2147483647  # Mersenne prime 2^31 - 1
        self.scale_factor = 1e6  # Scale factor for float to int conversion

        # Storage for protocol state
        self.pairwise_keys = {}  # [client_i][client_j] = shared_key
        self.ss_keys = {}  # Secret sharing keys for each client
        self.commitments = {}  # Public commitments for verification
        self.setup_phase()

    def setup_phase(self) -> Dict[int, Dict]:
        """
        Phase 1: Setup - Generate keys and commitments
        Returns: Dictionary with setup information for each client
        """
        client_setup_info = {}

        for client_i in range(self.total_clients):
            # Generate pairwise keys with all other clients
            pairwise_keys_i = {}
            for client_j in range(self.total_clients):
                if client_i != client_j:
                    # Generate shared key using deterministic method
                    # In practice, this would be done via Diffie-Hellman key exchange
                    shared_key = self._generate_shared_key(client_i, client_j)
                    pairwise_keys_i[client_j] = shared_key

            # Generate secret sharing key (random value)
            ss_key = random.randint(1, self.modulus_q - 1)

            # Store keys (in practice, each client would only know their own keys)
            if client_i not in self.pairwise_keys:
                self.pairwise_keys[client_i] = {}
            self.pairwise_keys[client_i] = pairwise_keys_i
            self.ss_keys[client_i] = ss_key

            # Create setup info for this client
            client_setup_info[client_i] = {
                "client_id": client_i,
                "pairwise_keys": pairwise_keys_i,
                "ss_key": ss_key,
                "threshold": self.threshold_t,
                "modulus": self.modulus_q,
            }

        return client_setup_info

    def _generate_shared_key(self, client_i: int, client_j: int) -> int:
        """
        Generate a shared key between two clients
        In practice, this would be done via Diffie-Hellman key exchange
        Here we use a deterministic method for simulation
        """
        # Ensure same key regardless of order (i,j) or (j,i)
        min_id, max_id = min(client_i, client_j), max(client_i, client_j)

        # Use HMAC for deterministic key generation
        key_material = f"client_{min_id}_client_{max_id}".encode()
        seed = b"secure_aggregation_seed_2024"  # In practice, use secure random seed

        hmac_obj = hmac.new(seed, key_material, hashlib.sha256)
        key_bytes = hmac_obj.digest()

        # Convert to integer in finite field
        key_int = int.from_bytes(key_bytes[:4], byteorder="big") % self.modulus_q
        return key_int

    def prg(self, seed: int, length: int) -> List[int]:
        """
        Pseudorandom generator - generates deterministic random sequence from seed
        """
        # Convert seed to bytes for hashing
        seed_bytes = seed.to_bytes(8, byteorder="big")
        output = []
        for i in range(length):
            # Generate hash for each position
            h = hashlib.sha256(seed_bytes + i.to_bytes(4, byteorder="big")).digest()
            # Convert first 4 bytes to integer in finite field
            value = int.from_bytes(h[:4], byteorder="big") % self.modulus_q
            output.append(value)

        return output

    def generate_mask(self, param_shape, param_len: int) -> torch.Tensor:
        """
        Generate random mask for a client using pairwise keys
        """
        mask = torch.zeros(size=param_shape).to(self.device).flatten()
        for c_id in range(self.total_clients):
            if c_id != self.client_id and c_id in self.pairwise_keys[self.client_id]:
                shared_key = self.pairwise_keys[self.client_id][c_id]
                # Generate pseudorandom values using the shared key
                prg_output = self.prg(seed=shared_key, length=param_len)

                # Add or subtract based on client ID ordering (ensures cancellation)
                if self.client_id < c_id:
                    # Add contribution
                    for i in range(param_len):
                        mask[i] = (mask[i] + prg_output[i]) % self.modulus_q
                else:
                    # Subtract contribution
                    for i in range(param_len):
                        mask[i] = (mask[i] - prg_output[i]) % self.modulus_q

        return torch.reshape(mask, param_shape)

    def finite_field_to_float(self, ff_value_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert finite field values (torch.Tensor) back to floats,
        handling negative values correctly.
        """
        # Create a boolean tensor representing the condition for negative numbers
        is_negative = ff_value_tensor > (self.modulus_q // 2)

        # Calculate the positive values (directly scaled)
        positive_values = ff_value_tensor / self.scale_factor

        # Calculate the negative values (converted and then scaled)
        negative_values = -(self.modulus_q - ff_value_tensor) / self.scale_factor

        # Use torch.where to select between positive and negative values based on the condition
        float_value_tensor = torch.where(is_negative, negative_values, positive_values)

        return float_value_tensor

    def broadcast_model(self, model) -> torch.nn.Module:
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self, epoch) -> None:
        epoch_strt = perf_counter_ns()
        for inputs, labels in self.train_data:
            itr_strt = perf_counter_ns()
            init_time = perf_counter_ns()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)
            loss.backward()
            compute_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            # compute original gradients' norm after aggregation
            # with torch.no_grad():
            #     duplicate_model = copy.deepcopy(self.model)
            #     for (name1, param1), (name2, param2) in zip(self.model.named_parameters(), duplicate_model.named_parameters()):
            #         param2.data = param1.data
            #         param2.grad = param1.grad
            #
            #     self.communicator.aggregate(msg=duplicate_model, communicate_params=False, compute_mean=False)
            #     og_grad_norm = compute_gradient_norm(model=duplicate_model)
            #     print(f'original aggregated grad_norm on client-{self.client_id}: {og_grad_norm}')

            init_time = perf_counter_ns()
            # scaled_gradients = []
            masked_grad = {}
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    scaled_grad = (param.grad * self.scale_factor).int()
                    # if scaled_grad < 0:
                    #     scaled_grad = (
                    #         scaled_grad % self.modulus_q + self.modulus_q
                    #     ) % self.modulus_q
                    # else:
                    #     scaled_grad = scaled_grad % self.modulus_q
                    # # scaled_gradients.append(scaled_grad)

                    # # Create a boolean tensor where True indicates scaled_grad is negative
                    condition = scaled_grad < 0
                    # Calculate the modulus for negative and non-negative values separately
                    negative_mod = (torch.remainder(scaled_grad, self.modulus_q) + self.modulus_q) % self.modulus_q
                    positive_mod = torch.remainder(scaled_grad, self.modulus_q)
                    # Use torch.where to choose between the two results based on the condition
                    scaled_grad = torch.where(condition, negative_mod, positive_mod)

                    # currently computed at every iteration, move later to run ONLY ONCE!
                    mask = self.generate_mask(
                        param_shape=scaled_grad.shape, param_len=scaled_grad.numel()
                    )

                    # apply mask
                    masked_grad[name] = (scaled_grad + mask) % self.modulus_q

            sec_agg_mask_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            init_time = perf_counter_ns()
            # now aggregate masked grads
            aggregated_masks = self.communicator.secure_aggregate(
                masked_grad=masked_grad, modulus_q=self.modulus_q
            )
            del masked_grad
            sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            with torch.no_grad():
                for (name1, param), (name2, agg_mask_grad) in zip(
                    self.model.named_parameters(), aggregated_masks.items()
                ):
                    assert name1 == name2
                    param.grad.copy_(self.finite_field_to_float(ff_value_tensor=agg_mask_grad))

            # compute unmasked, aggregated gradients' norm
            # with torch.no_grad():
            #     masked_agg_grad_norm = compute_gradient_norm(model=self.model)
            #     print(f"secure_aggregation grad_norm on client-{self.client_id}: {masked_agg_grad_norm}")

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            itr_time = (perf_counter_ns() - itr_strt) / nanosec_to_millisec
            logging.info(
                f"training_metrics local_step: {self.local_step} epoch {epoch} compute_time {compute_time} ms "
                f"sec_agg_mask_time {sec_agg_mask_time} ms sync_time {sync_time} ms itr_time: {itr_time} ms"
            )
            # del duplicate_model

        epoch_time = (perf_counter_ns() - epoch_strt) / nanosec_to_millisec
        logging.info(f"epoch completion time for epoch {epoch} is {epoch_time} ms")

        train_img_accuracy(
            epoch=epoch,
            iteration=self.local_step,
            input=inputs,
            label=labels,
            output=pred,
            loss=loss,
            train_loss=self.train_loss,
            top1acc=self.top1_acc,
            top5acc=self.top5_acc,
            top10acc=self.top10_acc,
        )
        test_img_accuracy(
            epoch=epoch,
            device=self.device,
            model=self.model,
            test_loader=self.test_data,
            loss_fn=self.loss,
            iteration=self.local_step,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train(self):
        print("RUNNING SIMULATED SECURE AGGREGATION...")
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                print("going to start epoch {}/{}".format(epoch, self.epochs))
                self.train_loop(epoch=epoch)
        else:
            i = 0
            while True:
                self.train_loop(epoch=i)
                i += 1
