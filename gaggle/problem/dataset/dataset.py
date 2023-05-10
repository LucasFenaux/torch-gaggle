import copy
from abc import ABC
from copy import deepcopy
from pprint import pprint
from typing import List, Union

import numpy as np
import torch.utils.data
from torch.utils.data.dataset import T_co
from torchvision import transforms
import torchvision
from gaggle.arguments.problem_args import ProblemArgs
from gaggle.arguments.sys_args import SysArgs
from gaggle.utils.special_images import plot_images


class DataWrapper(torch.utils.data.Dataset):
    """Wrapper that set the .data and .targets attributes that can then be accessed by the Dataset class in the
    get_data_and_targets method.

    See Also:
        This class creates attributes that are used by Dataset.get_data_and_targets.

    """
    def __init__(self, data: torch.Tensor = None, targets: torch.Tensor = None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.targets[item]
        return x, y


class Dataset(torch.utils.data.Dataset, ABC):
    r"""Dataset class that allows for more flexible custom indexing and other behavior"""
    def __init__(self, problem_args: ProblemArgs = None, train: bool = True, sys_args: SysArgs = None):
        self.train = train
        super(Dataset, self).__init__()
        self.problem_args = problem_args if problem_args is not None else ProblemArgs()
        self.sys_args = sys_args if sys_args is not None else SysArgs()
        self.apply_transform: bool = True

        # pre-processors receive an index, the image and the label of each item
        self.idx: List[int] = []  # all indices that this dataset returns
        self.idx_to_backdoor = {}  # store idx and backdoor mapping

        self.dataset: torch.utils.data.Dataset | None = None
        self.classes: List[str] = []
        self.real_normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.class_to_idx = None
        self.disable_fetching = False

    def get_data_and_targets(self):
        r"""Gets the data and the targets for the current dataset stored in the self.data object.
        The self.data object should have .data and .targets attributes to be returned.

        Returns:
            A tuple containing (data, targets) or (None, None) if the dataset is not initialized.

        """
        if self.dataset is not None:
            return (self.dataset.data, self.dataset.targets)
        else:
            return (None, None)

    def get_data_and_transform(self):
        """

        Returns:
            Returns ((data, targets), transforms)
        """
        out_transform = copy.deepcopy(self.transform)
        # remove the to_tensor transform as well already be working with tensors
        new_transforms = []
        for transform in self.transform.transforms:
            if isinstance(transform, torchvision.transforms.ToTensor):
                continue
            new_transforms.append(transform)
        out_transform.transforms = new_transforms
        data = self.get_data_and_targets()
        return data, out_transform

    def num_classes(self) -> int:
        """ Return the number of classes"""
        return len(self.classes)

    def _build_transform(self) -> None:
        """ Internal function to build a default transformation. Override this
        if necessary. """
        transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])
        return transform

    def random_subset(self, n: int):
        """ Creates a random subset of this dataset """
        idx = deepcopy(self.idx)
        np.random.shuffle(idx)

        copy = self.copy()
        copy.idx = idx[:n]
        copy.class_to_idx = None
        return copy

    def subset(self, idx: Union[List[int], int]):
        """ Creates a subset of this dataset. """
        if isinstance(idx, int):
            idx = np.arange(idx)
        copy = self.copy()
        copy.idx = [self.idx[i] for i in idx]
        copy.class_to_idx = None
        return copy

    def remove_classes(self, target_classes: List[int]):
        """ Creates a subset without samples from one target class. """
        copy = self.copy()
        for target_class in target_classes:
            for index_of_target_class in copy.get_class_to_idx()[target_class]:
                copy.idx.remove(index_of_target_class)
        copy.class_to_idx = None
        return copy

    def visualize(self, sq_size: int = 3) -> None:
        """ Plot samples from this dataset as a square.
        """
        n = sq_size ** 2
        x = [self[i][0] for i in range(n)]
        imgs = torch.stack(x, 0)

        if len(self.idx_to_backdoor) == 0:
            title = self.problem_args.dataset_name
        else:
            title = f"{self.problem_args.dataset_name} (Poisoned)"
        plot_images(imgs, n_row=sq_size, title=title)

    def print_class_distribution(self):
        class_to_idx = self.get_class_to_idx(verbose=False)
        cd = {c: 100 * len(v) / len(self) for c, v in class_to_idx.items()}
        pprint(cd)

    def without_normalization(self) -> 'Dataset':
        """ Return a copy of this data without normalization.
        """
        copy = self.copy()
        copy.enable_normalization(False)
        return copy

    def enable_normalization(self, enable: bool) -> None:
        """ Method to enable or disable normalization.
        """
        if enable:
            self.normalize_transform = self.real_normalize_transform
        else:
            self.normalize_transform = transforms.Lambda(lambda x: x)
        self.transform = self._build_transform()

    def size(self):
        """ Alternative function to get the size. """
        return len(self.idx)

    def __len__(self):
        """ Return the number of elements in this dataset """
        return self.size()

    def copy(self):
        """ Return a copy of this dataset instance. """
        return deepcopy(self)

    def __getitem__(self, index) -> T_co:
        index = self.idx[index]
        x, y = self.dataset[index]
        x = self.transform(x)
        return self.normalize(x), y
