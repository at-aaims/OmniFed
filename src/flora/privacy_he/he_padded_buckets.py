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

import logging
from time import perf_counter_ns

import torch
import tenseal as ts

from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.flora.helper.training_params import FedAvgTrainingParameters
from src.flora.privacy.homomorphic_encryption import HomomorphicEncryption
import src.flora.privacy.homomorphic_encryption as he_utils
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)

nanosec_to_millisec = 1e6


class HEPaddedBuckets:
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        communicator: TorchMPICommunicator,
        total_clients: int,
        train_params: FedAvgTrainingParameters,
        poly_modulus_degree: int,
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

        self.encrypt_grads = True
        self.poly_modulus_degree = poly_modulus_degree
        self.he_buckets = he_utils.HomomorphicEncryptBucketing(
            poly_modulus_degree=self.poly_modulus_degree
        )
        self.chunk_size = self.he_buckets.get_chunk_size()
        print("going to initiate seal context....")
        self.handle_he_ctx(poly_modulus_degree=self.poly_modulus_degree)
        print("@@@@@@@@@@@@@@@@@@@@@@@@ created HE context @@@@@@@@@@@@@@@@@@@@@@@@@@")

    def handle_he_ctx(self, poly_modulus_degree):
        """
        compute HE context on rank 0 and send serialized context to clients
        """
        if self.client_id == 0:
            self.he_obj = HomomorphicEncryption(
                poly_modulus_degree=poly_modulus_degree, encrypt_grads=True
            )
            serialized_ctx = self.he_obj.get_he_context().serialize(
                save_secret_key=False
            )
            ctx_bytes = torch.ByteTensor(list(serialized_ctx))
            ctx_size = torch.tensor([ctx_bytes.numel()], dtype=torch.long)
            print("done on client-0.....")
        else:
            ctx_size = torch.zeros(1, dtype=torch.long)
            print("done on client-", self.client_id)

        ctx_size = self.communicator.broadcast(msg=ctx_size, id=0)
        print("context_size: ", ctx_size.item(), "on client", self.client_id)
        ctx_buf = torch.empty(ctx_size.item(), dtype=torch.uint8)

        if self.client_id == 0:
            ctx_buf[:] = ctx_bytes

        ctx_buf = self.communicator.broadcast(msg=ctx_buf, id=0)

        if self.client_id == 0:
            self.context = self.he_obj.get_he_context()
        else:
            serialized_ctx = bytes(ctx_buf.tolist())
            self.context = ts.context_from(serialized_ctx)

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self, epoch):
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

            # encrypt model updates into chunks
            init_time = perf_counter_ns()
            encrypted_data = []

            for param in self.model.parameters():
                if param.grad is None:
                    continue

                chunks = self.he_buckets.chunk_tensors(
                    tensor=param.grad.detach().cpu(), chunk_size=self.chunk_size
                )
                enc_chunks = self.he_buckets.encrypt_chunks(
                    chunks=chunks, context=self.context
                )

                # Store both serialized data and original sizes
                param_data = []
                for chunk in enc_chunks:
                    serialized = chunk.serialize()
                    serialized_tensor = torch.ByteTensor(list(serialized))
                    param_data.append(serialized_tensor)

                encrypted_data.append(param_data)

            he_encryption_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            # Communicate encrypted chunks with proper size handling
            init_time = perf_counter_ns()
            aggregated_data = []

            for param_idx, param_chunks in enumerate(encrypted_data):
                param_aggregated = []

                for chunk_idx, chunk_data in enumerate(param_chunks):
                    # Step 1: Broadcast the size of this chunk to all workers
                    chunk_size = torch.tensor([chunk_data.size(0)], dtype=torch.long)
                    size_list = [
                        torch.zeros_like(chunk_size) for _ in range(self.total_clients)
                    ]

                    # dist.all_gather(tensor_list=size_list, tensor=chunk_size)
                    self.communicator.he_collect(recv_buff=size_list, msg=chunk_size)

                    # Find the maximum size
                    max_size = max(size.item() for size in size_list)

                    # Step 2: Pad this worker's chunk to the maximum size
                    if chunk_data.size(0) < max_size:
                        padding = torch.zeros(
                            max_size - chunk_data.size(0), dtype=torch.uint8
                        )
                        padded_chunk = torch.cat([chunk_data, padding])
                    else:
                        padded_chunk = chunk_data

                    # Step 3: Gather the padded chunks
                    recv_bufs = [
                        torch.empty_like(padded_chunk)
                        for _ in range(self.total_clients)
                    ]
                    # dist.all_gather(tensor_list=recv_bufs, tensor=padded_chunk)
                    self.communicator.he_collect(recv_buff=recv_bufs, msg=padded_chunk)

                    # Step 4: Store both the received data and the original sizes
                    chunk_info = {
                        "data": recv_bufs,
                        "original_sizes": [size.item() for size in size_list],
                    }
                    param_aggregated.append(chunk_info)

                aggregated_data.append(param_aggregated)

            encrypted_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            # CENTRALIZED DECRYPTION: Only rank 0 decrypts, then broadcasts results
            init_time = perf_counter_ns()

            if self.client_id == 0:
                # Rank 0 performs decryption and averaging
                decrypted_gradients = []

                for param_idx, param_chunks in enumerate(aggregated_data):
                    param_avg_grad = []

                    for chunk_info in param_chunks:
                        recv_data = chunk_info["data"]
                        original_sizes = chunk_info["original_sizes"]

                        # Deserialize each worker's contribution
                        vectors = []
                        for worker_idx, (data, orig_size) in enumerate(
                            zip(recv_data, original_sizes)
                        ):
                            # Truncate to original size before deserializing
                            truncated_data = data[:orig_size]
                            try:
                                vec = ts.ckks_vector_from(
                                    self.context, bytes(truncated_data.tolist())
                                )
                                vectors.append(vec)
                            except Exception as e:
                                print(
                                    f"Rank 0: Failed to deserialize chunk from worker {worker_idx}: {e}"
                                )
                                raise e

                        # Average the vectors
                        if vectors:
                            avg_vector = vectors[0]
                            for vec in vectors[1:]:
                                avg_vector += vec
                            avg_vector *= 1.0 / self.total_clients

                            # Decrypt (only rank 0 can do this)
                            decrypted_chunk = avg_vector.decrypt()
                            param_avg_grad.extend(decrypted_chunk)

                    decrypted_gradients.append(param_avg_grad)

                # Convert to tensors for broadcasting
                gradient_tensors = []
                param_shapes = []
                param_idx = 0

                for param in self.model.parameters():
                    if param.grad is None:
                        gradient_tensors.append(None)
                        param_shapes.append(None)
                        continue

                    if param_idx < len(decrypted_gradients):
                        avg_grad = decrypted_gradients[param_idx]
                        if avg_grad:
                            avg_grad_tensor = torch.tensor(
                                avg_grad[: param.numel()], dtype=torch.float32
                            )
                            gradient_tensors.append(avg_grad_tensor)
                            param_shapes.append(param.shape)
                        else:
                            gradient_tensors.append(
                                torch.zeros(param.numel(), dtype=torch.float32)
                            )
                            param_shapes.append(param.shape)
                        param_idx += 1
                    else:
                        gradient_tensors.append(
                            torch.zeros(param.numel(), dtype=torch.float32)
                        )
                        param_shapes.append(param.shape)

            else:
                # Non-root ranks prepare empty tensors to receive broadcasts
                gradient_tensors = []
                param_shapes = []

                for param in self.model.parameters():
                    if param.grad is None:
                        gradient_tensors.append(None)
                        param_shapes.append(None)
                    else:
                        gradient_tensors.append(
                            torch.zeros(param.numel(), dtype=torch.float32)
                        )
                        param_shapes.append(param.shape)

            for i, (grad_tensor, shape) in enumerate(
                zip(gradient_tensors, param_shapes)
            ):
                if grad_tensor is not None:
                    # dist.broadcast(tensor=grad_tensor, src=0)
                    self.communicator.broadcast(msg=grad_tensor, id=0)

            # Apply the broadcasted gradients
            with torch.no_grad():
                param_idx = 0
                for param in self.model.parameters():
                    if param.grad is None:
                        continue

                    if (
                        param_idx < len(gradient_tensors)
                        and gradient_tensors[param_idx] is not None
                    ):
                        grad_tensor = (
                            gradient_tensors[param_idx]
                            .reshape_as(param)
                            .to(self.device)
                        )
                        param.grad.copy_(grad_tensor)

                    param_idx += 1

            he_decryption_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            itr_time = (perf_counter_ns() - itr_strt) / nanosec_to_millisec

            logging.info(
                f"training_metrics local_step: {self.local_step} epoch {epoch} compute_time {compute_time} ms "
                f"he_encryption_time: {he_encryption_time} ms he_decryption_time {he_decryption_time} ms "
                f"encrypted_sync_time: {encrypted_sync_time} ms itr_time: {itr_time} ms"
            )

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
        print("going to broadcast model across clients...")
        self.model = self.broadcast_model(model=self.model)
        print(
            f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ BROADCAST MODEL COMPLETE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        )
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                print("going to start epoch {}/{}".format(epoch, self.epochs))
                self.train_loop(epoch=epoch)
        else:
            i = 0
            while True:
                self.train_loop(epoch=i)
                i += 1
