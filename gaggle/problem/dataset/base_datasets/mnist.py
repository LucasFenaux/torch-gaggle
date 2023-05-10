import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from gaggle.arguments.problem_args import ProblemArgs
from gaggle.arguments.sys_args import SysArgs
from gaggle.problem.dataset.dataset import Dataset
from gaggle.global_configs import global_configs


class MNIST(Dataset):
    """MNIST dataset. Deng, L. (2012). The mnist database of handwritten digit images for machine learning research.
    IEEE Signal Processing Magazine, 29(6), 141–142.

    """
    def __init__(self, problem_args: ProblemArgs = None, train: bool = True, sys_args: SysArgs = None):
        super().__init__(problem_args, train, sys_args)
        self.dataset = torchvision.datasets.MNIST(root=global_configs.CACHE_DIR, download=True, train=train, transform=None)
        self.idx = list(range(len(self.dataset)))

        max_size = self.problem_args.max_size_train if train else self.problem_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)

        self.transform = self._build_transform()
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def get_data_and_targets(self):
        if isinstance(self.dataset.data, torch.Tensor):
            data = (self.dataset.data.to(torch.float), self.dataset.targets.to(torch.float))
        else:
            data = (torch.Tensor(self.dataset.data), torch.Tensor(self.dataset.targets))
        return data

    def _build_transform(self):
        if self.train:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        return transform
