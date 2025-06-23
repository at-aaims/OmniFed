import rich.repr
import torch
import torch.utils.data as td

# ======================================================================================


@rich.repr.auto
class DummyDataset(td.TensorDataset):
    """
    Synthetic dataset generator for federated learning experimentation and testing.
    """

    def __init__(
        self, num_samples: int = 200, input_dim: int = 10, num_classes: int = 2
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))
        super().__init__(X, y)
