from typing import Callable
from gaggle.arguments.ga_args import GAArgs
from gaggle.operators.crossover.crossover import Crossover
from gaggle.operators.crossover.base_crossovers.uniform_crossover import UniformCrossover
from gaggle.operators.crossover.base_crossovers.k_point_crossover import KPointCrossover


class CrossoverFactory:
    r"""Factory that generates pre-existing available crossover operators.
    CrossoverFactory.crossovers stores said crossover operators as a dictionary with 
    their name as the key and the uninitialized crossover object as the value.

    """
    crossovers = {
        "uniform": UniformCrossover,
        "k_point": KPointCrossover
    }

    @classmethod
    def get_keys(cls):
        r"""Returns the list of currently registered crossovers """
        return list(cls.crossovers.keys())

    @classmethod
    def update(cls, key, crossover):
        r"""Add a new crossover operator to the dictionary of crossovers that can be created.

        It is added to CrossoverFactory.crossovers

        Args:
            key: crossover name that will be used as the dictionary lookup key
            crossover: Crossover class object, it needs to not be already initialized

        """
        assert isinstance(crossover, Callable)
        cls.crossovers[key] = crossover

    @classmethod
    def from_ga_args(cls, ga_args: GAArgs = None) -> Crossover:
        r"""Initializes the requested crossover from the dictionary of available crossovers.

        This is done by using the attribute ga_args.crossover as
        the lookup key to CrossoverFactory.crossovers.

        Args:
            ga_args: GAArgs object for the current run

        Returns:
            An initialized Crossover class object.

        """
        ga_args = ga_args if ga_args is not None else GAArgs()
        crossover = cls.crossovers.get(ga_args.crossover, None)
        if crossover is None:
            raise ValueError(ga_args.crossover)
        return crossover(ga_args)
