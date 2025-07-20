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

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
import tenseal as ts

# Configuration
CHUNK_SIZE = 8192

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def create_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    return ctx

def chunk_tensor(tensor, chunk_size):
    flat = tensor.view(-1).tolist()
    return [flat[i:i + chunk_size] for i in range(0, len(flat), chunk_size)]

def encrypt_chunks(chunks, ctx):
    return [ts.ckks_vector(ctx, c) for c in chunks]

def decrypt_chunks(enc_chunks):
    return [c.decrypt() for c in enc_chunks]

def average_encrypted_chunks(chunks_list, world_size):
    avg_chunks = []
    for chunk_group in zip(*chunks_list):
        avg = chunk_group[0]
        for other in chunk_group[1:]:
            avg += other
        avg *= (1.0 / world_size)
        avg_chunks.append(avg)
    return avg_chunks

def train(rank, world_size):
    setup(rank, world_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    model = models.resnet18(num_classes=10).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    ctx = create_context()

    model.train()
    for epoch in range(1):
        for batch, (x, y) in enumerate(loader):
            x, y = x.to(rank), y.to(rank)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()

            enc_param_chunks_list = []
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach().cpu()
                chunks = chunk_tensor(grad, CHUNK_SIZE)
                enc_chunks = encrypt_chunks(chunks, ctx)
                serialized = [torch.ByteTensor(bytes(e.serialize())) for e in enc_chunks]
                enc_param_chunks_list.append(serialized)

            # Communicate all chunks across workers
            all_chunks_per_param = [[] for _ in enc_param_chunks_list]
            for i, param_chunks in enumerate(enc_param_chunks_list):
                for j, chunk in enumerate(param_chunks):
                    recv_bufs = [torch.empty_like(chunk) for _ in range(world_size)]
                    dist.all_gather(recv_bufs, chunk)
                    all_chunks_per_param[i].append(recv_bufs)

            # Deserialize and decrypt
            with torch.no_grad():
                for param, all_chunks in zip(model.parameters(), all_chunks_per_param):
                    if param.grad is None:
                        continue
                    avg_grad = []
                    for chunk_set in all_chunks:
                        vectors = [ts.ck]()
