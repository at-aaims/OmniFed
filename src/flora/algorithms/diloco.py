import torch
import copy

from src.flora.communicator import Communicator
from src.flora.helper.training_params import DiLocoTrainingParameters
from src.flora.helper.node_config import NodeConfig


class DiLoCo:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        id: int,
        total_clients: int,
        train_params: DiLocoTrainingParameters,
    ):
        self.model = model
        self.train_data = train_data
        self.communicator = communicator
        self.id = id
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.epochs = self.train_params.get_epochs()
        self.outer_lr = self.train_params.get_outer_lr()
        self.outer_momentum = self.train_params.get_outer_momentum()
        self.local_step = 0

        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.diff_params = copy.deepcopy(self.model)
        self.global_model, self.diff_params = (
            self.global_model.to(self.device),
            self.diff_params.to(self.device),
        )
        self.velocity = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }

    def initialize_model(self):
        # model broadcasted from central server with id 0
        self.model = self.communicator.broadcast(msg=self.model, id=0)

    def aggregate_updates(self):
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.global_model.parameters(), self.model.parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                target_param = dict(self.diff_params.named_parameters())[name1]
                target_param.copy_(param1 - param2)

        self.diff_params = self.communicator.aggregate(
            msg=self.diff_params, communicate_params=True, compute_mean=True
        )

    def _zero_velocity(self):
        for v in self.velocity.values():
            v.zero_()

    def _outer_step(self):
        with torch.no_grad():
            for (name, param), (_, param_delta) in zip(
                self.global_model.named_parameters(),
                self.diff_params.named_parameters(),
            ):
                v = self.velocity[name]
                # Momentum update rule
                v.mul_(self.outer_momentum).add_(param_delta.data, alpha=self.outer_lr)
                # Update model parameters
                param.data.add_(v)

        return self.global_model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            if self.local_step % self.comm_freq == 0:
                self.aggregate_updates()
                self.global_model = self._outer_step()
                self.model = self.global_model

    def train(self):
        self.initialize_model()
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()
