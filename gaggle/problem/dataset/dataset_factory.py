from typing import Callable

from gaggle.arguments.problem_args import ProblemArgs
from gaggle.arguments.sys_args import SysArgs
from gaggle.problem.dataset.base_datasets.cifar10 import CIFAR10
from gaggle.problem.dataset.base_datasets.mnist import MNIST
from gaggle.problem.dataset.dataset import Dataset, DataWrapper

import torch


class DatasetFactory:
    r"""Factory that generates pre-existing available datasets.
    DatasetFactory.datasets stores said datasets as a dictionary with their name as key and the uninitialized Dataset
    object as value.

    See Also:
        Dataset Class
    """
    datasets = {
        "CIFAR10": CIFAR10,
        "MNIST": MNIST,
    }

    @classmethod
    def get_keys(cls):
        r"""Gets the keys (dataset names) for the available pre-built datasets.

        Returns:
            list of strings that are the keys to DatasetFactory.datasets

        """
        return list(cls.datasets.keys())

    @classmethod
    def update(cls, key, dataset):
        r"""Add a new dataset to the dictionary of datasets that can be created.

        It is added to DatasetFactory.datasets

        Args:
            key: dataset name that will be used as the dictionary lookup key
            dataset: dataset class object, it needs to not be already initialized

        """
        assert isinstance(dataset, Callable)
        cls.datasets[key] = dataset

    @classmethod
    def from_problem_args(cls, problem_args: ProblemArgs = None, train: bool = True, sys_args: SysArgs = None) \
            -> Dataset:
        r"""Initializes the requested dataset from the dictionary of available datasets.

        This is done by using the attribute problem_args.dataset_name as
        the lookup key to DatasetFactory.datasets.

        Args:
            problem_args: problem args that will be used to build the Dataset
            train: whether we should return the training or evaluation dataset
            sys_args: system args

        Returns:
            A Dataset object.

        """
        problem_args = problem_args if problem_args is not None else ProblemArgs()
        dataset = cls.datasets.get(problem_args.problem_name, None)
        if dataset is None:
            raise ValueError(problem_args.problem_name)
        return dataset(problem_args, train=train, sys_args=sys_args)

    @staticmethod
    def from_data(data: torch.Tensor, targets: torch.Tensor, train: bool = True, seed: int = 1337) -> Dataset:
        r"""Creates a basic dataset object from given data and targets with basic arguments.

        Args:
            data: data tensor
            targets: target/label tensor
            train: whether it is a training or evaluation dataset
            seed: seed for the randomness of the batch sampling

        Returns:
            A Dataset object.

        """
        # using default dataset args:
        problem_args = ProblemArgs()
        problem_args.dataset_name = "custom"
        problem_args.seed = seed

        dataset = Dataset(problem_args=problem_args, train=train)
        dataset.dataset = DataWrapper(data=data, targets=targets)
        return dataset
