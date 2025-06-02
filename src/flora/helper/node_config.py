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

import subprocess

import psutil


def nvdia_gpu_count():
    try:
        # Run the `nvidia-smi` command and capture the output
        output = subprocess.check_output(["nvidia-smi", "-L"])
        gpu_count = len(output.decode().strip().split("\n"))
        return gpu_count
    except Exception as e:
        # Return 0 if no NVIDIA GPUs are found or an error occurs
        return 0


def amd_gpu_count():
    try:
        # Run the `lspci` command and capture the output and filter for AMD GPUs
        output = subprocess.check_output(["lspci"])
        amd_gpu_count = sum(
            "Advanced Micro Devices, Inc. [AMD/ATI]" in line
            for line in output.decode().strip().split("\n")
        )
        return amd_gpu_count
    except Exception as e:
        # Return 0 if no AMD GPUs are found or an error occurs
        return 0


def detect_gpus_if_any():
    nvidia_count = nvdia_gpu_count()
    amd_count = amd_gpu_count()

    if nvidia_count > 0:
        return nvidia_count
    elif amd_count > 0:
        return amd_count
    else:
        return 0


def get_system_memory():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    # return total available memory in MBs
    return total_memory / (1024 * 1024)


class NodeConfig(object):
    def __init__(self):
        self.total_system_mem = get_system_memory()
        self.total_cpus = psutil.cpu_count(logical=True)
        self.total_gpus = detect_gpus_if_any()

    def get_total_system_mem(self):
        return self.total_system_mem

    def get_cpus(self):
        return self.total_cpus

    def get_gpus(self):
        return self.total_gpus
