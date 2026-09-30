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

"""FedSGD is FedAvg with gradient payload — same loop, no extra math here."""

import rich.repr

from .fedavg import FedAvg


@rich.repr.auto
class FedSGD(FedAvg):
    """Named FedSGD handle for Hydra (``algorithm: fedsgd``).

    Training, sample-weighted averaging, and when to ``optimizer.step`` live in
    ``BaseAlgorithm``. Defaults come from ``conf/algorithm/fedsgd.yaml``:
    ``aggregate_payload: gradients`` and ``batch_end.every: 1``.
    """
