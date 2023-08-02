import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from gaggle.arguments.problem_args import ProblemArgs
from gaggle.arguments.sys_args import SysArgs
from gaggle.problem.dataset.dataset import Dataset
from gaggle.global_configs import global_configs


class CIFAR10(Dataset):
    """CIFAR10 dataset. Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

    """

    def __init__(self, problem_args: ProblemArgs, train: bool = True, sys_args: SysArgs = None):
        super().__init__(problem_args, train, sys_args)
        self.dataset = torchvision.datasets.CIFAR10(root=global_configs.CACHE_DIR, download=True, train=train,
                                                    transform=torchvision.transforms.ToTensor())
        self.idx = list(range(len(self.dataset)))

        self.real_normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.normalize_transform = self.real_normalize_transform

        max_size = self.problem_args.max_size_train if train else self.problem_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)
        self.transform = self._build_transform()
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def get_data_and_targets(self):
        targets = []
        images = []
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            images.append(x)
            targets.append(y)
        return torch.stack(images), torch.Tensor(targets)

    def _build_transform(self):
        if self.train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform = transforms.Compose([
            ])
        return transform
