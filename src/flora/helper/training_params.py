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

import torch
import torch.nn as nn


class TrainingParameters(object):
    def __init__(self, **kwargs):
        self.optimizer = kwargs.get("optimizer", None)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.loss = kwargs.get("loss", None)
        self.epochs = kwargs.get("epochs", None)


class MLPModel(nn.Module):
    """basic fully-connected network"""

    def __init__(self, grad_dim, hidden_dim=1024, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(grad_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, grad_dim),
        )

    def forward(self, large_batch_update):
        return self.net(large_batch_update)


if __name__ == "__main__":
    model = MLPModel(grad_dim=10)
    optimer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    comm_freq = 50

    params = TrainingParameters(optimizer=optimer, comm_freq=comm_freq, loss=loss)
    print(params.epochs)
