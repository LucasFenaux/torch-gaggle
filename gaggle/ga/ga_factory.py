from typing import Callable
from gaggle.arguments import GAArgs, SysArgs, OutdirArgs, IndividualArgs, ProblemArgs
from gaggle.ga import GA, SimpleGA
from gaggle.operators import Selection, Mutation, Crossover
from gaggle.population import PopulationManager
from gaggle.problem import Problem


class GAFactory:
    r"""Factory that generates pre-existing available GA algorithms.
    GAFactory.gas stores said GAs as a dictionary with their name as key and the uninitialized GA
    object as value.
    """
    gas = {
        "simple": SimpleGA,
    }

    @classmethod
    def update(cls, key, ga):
        r"""Add a new GA to the dictionary of GAs that can be created.

        It is added to GAFactory.gas

        Args:
            key: dataset name that will be used as the dictionary lookup key
            ga: GA class object, it needs to not be already initialized

        """
        assert isinstance(ga, Callable)
        cls.gas[key] = ga

    @classmethod
    def from_ga_args(cls, population_manager: PopulationManager = None, ga_args: GAArgs = None,
                     selection: Selection = None, crossover: Crossover = None, mutation: Mutation = None,
                     problem_args: ProblemArgs = None, sys_args: SysArgs = None, outdir_args: OutdirArgs = None,
                     individual_args: IndividualArgs = None, problem: Problem = None) -> GA:
        r"""Initializes the requested GA from the dictionary of available GAs.

        This is done by using the attribute ga_args.ga_name as
        the lookup key to GAFactory.gas.

        Args:
            population_manager:
            ga_args:
            selection:
            crossover:
            mutation:
            problem_args:
            sys_args:
            outdir_args:
            individual_args:
            problem:

        Returns:
            An initialized GA class object.

        """
        ga_args = ga_args if ga_args is not None else GAArgs()
        ga = cls.gas.get(ga_args.ga_name, None)
        if ga is None:
            raise ValueError(ga_args.ga_name)
        return ga(population_manager=population_manager, ga_args=ga_args, outdir_args=outdir_args, sys_args=sys_args,
                  individual_args=individual_args, problem_args=problem_args, selection=selection, mutation=mutation,
                  crossover=crossover, problem=problem)
