import copy
import random
from abc import abstractmethod

from gaggle.population.individual import Individual
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager


class Crossover:
    r""" The parent class for any Crossover Operator. 
    It gives a basic function to crossover a whole population once the function for crossing over a single pair of parents is specified

    """
    mates_per_crossover = 0
    children_per_crossover = 0

    def __init__(self, ga_args: GAArgs = None):
        self.ga_args = ga_args if ga_args is not None else GAArgs()


    @abstractmethod
    def crossover_individual(self, individuals: list[Individual]) -> list[Individual]:
        r"""Speficies how to create children from parents
        Args:
            individuals: a list of parents to crossover (typically 2)
        Returns:
            A list of children created from the parents (typically 2)"""
        raise NotImplementedError

    def crossover_pop(self, manager: PopulationManager) -> PopulationManager:
        r""" Calls the crossover indivual operator over the whole popualtion
        while maintaining the protected parents
        For each pair of parents, crossover is called with probability ga_args.parent_survival_rate
        Args:
            manager: PopulationManager object holding the current population
        Returns:
            Modified PopulationManager object"""
        parents = manager.get_parents()
        num_parents = len(parents)
        population = manager.get_population()
        freshness = manager.get_freshness()

        new_population = {}
        new_freshness = {}

        free_indices = list(range(self.ga_args.population_size))
        to_mutate = []

        # first we get the protected indices and we add them to the list
        protected_models = manager.get_protected()

        # adding the protected
        for p in protected_models:
            new_population[p] = population[p]
            new_freshness[p] = freshness[p]
            free_indices.remove(p)
            if self.ga_args.mutate_protected:
                to_mutate.append(p)

        mating_tuples = manager.get_mating_tuples()
        surviving_parents = []
        to_mate = []
        # now we count how many parents to keep
        for mating_tuple in mating_tuples:
            # probability to keep the parents rather than the children
            keep_parents = random.random() < self.ga_args.parent_survival_rate
            if keep_parents:
                surviving_parents.extend(mating_tuple)
            else:
                to_mate.append(mating_tuple)

        # we now fill the lucky parents that survived
        for idx in surviving_parents:
            if idx in free_indices:
                new_population[idx] = copy.deepcopy(population[idx])  # we might have put that model already in so we don't want a double reference
                new_freshness[idx] = freshness[idx]
                free_indices.remove(idx)
                to_mutate.append(idx)
            else:
                for new_idx in free_indices:
                    if new_idx in surviving_parents:
                        continue
                    new_population[new_idx] = copy.deepcopy(population[idx])  # we might have put that model already in so we don't want a double reference
                    new_freshness[new_idx] = freshness[idx]
                    free_indices.remove(new_idx)
                    to_mutate.append(new_idx)
                    break

        # now we can do the actual crossover
        for mating_tuple in to_mate:
            party = [manager.population[idx] for idx in mating_tuple]
            children = self.crossover_individual(copy.deepcopy(party))
            for child in children:
                new_idx = free_indices.pop(0)
                to_mutate.append(new_idx)
                new_population[new_idx] = child
                new_freshness[new_idx] = True

        # if we are missing some, we fill the population
        for i in free_indices:
            idx = parents[random.randint(0, num_parents - 1)]
            new_population[i] = copy.deepcopy(population[idx])  # we might have put that model already in so we don't want a double reference
            new_freshness[i] = freshness[idx]
            free_indices.remove(i)
            to_mutate.append(i)

        # we finally update the manager
        manager.update_population(new_population, new_freshness)
        manager.update_to_mutate(new_to_mutate=to_mutate)
        return manager
