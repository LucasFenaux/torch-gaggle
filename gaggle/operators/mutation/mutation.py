from abc import abstractmethod

from gaggle.population.individual import Individual
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager


class Mutation:
    r""" The parent class for any Mutation Operator. 
    It gives a basic function to mutate a whole population once the function for mutating a single individual is specified

    """
    def __init__(self, ga_args: GAArgs = None):
        self.ga_args = ga_args if ga_args is not None else GAArgs()

    @abstractmethod
    def mutate_individual(self, individual: Individual) -> Individual:
        r"""Speficies how to mutate a single individual
        Args:
            individuals: a single individual to mutate
        Returns:
            A single individual after mutation"""
        raise NotImplementedError

    def mutate_pop(self, manager: PopulationManager) -> PopulationManager:
        r""" Calls the mutate_individual function for each member of the population
        Args:
            manager: PopulationManager object holding the current population
        Returns:
            Modified PopulationManager object"""
        population = manager.get_population()
        new_freshness = manager.get_freshness()

        to_mutate = manager.get_to_mutate()
        for individual_idx in to_mutate:
            population[individual_idx] = self.mutate_individual(population[individual_idx])
            new_freshness[individual_idx] = True

        manager.update_population(population, new_freshness)
        # in case we need to enforce parameter value bounds on the freshly mutated samples. Since it only applies to
        # mutated samples we don't have to update the freshness
        manager.apply_bounds(to_mutate)
        return manager
