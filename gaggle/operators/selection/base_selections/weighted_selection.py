import copy

import numpy as np
from numpy.random import choice

from gaggle.operators.selection.selection import Selection
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager


class WeightedSelection(Selection):
    r"""Standard Roulette Wheel selection
    Probability of selection is fittness/total fittness
    If negative fittness all fittness values are shifted to make positive
    
    """
    def __init__(self, ga_args: GAArgs = None):
        super(WeightedSelection, self).__init__(ga_args=ga_args)

    def select_parents(self, manager: PopulationManager, mates_per_crossover: int, children_per_crossover: int) -> PopulationManager:
        fitness = copy.deepcopy(manager.get_fitness())
        min_fit = min(fitness.items(), key=lambda x: x[1])[1]
        if min_fit < 0:
            offset = np.abs(min_fit)
            for key in fitness.keys():
                fitness[key] += offset
        p = []
        fitness_sum = 0.
        for key in fitness.keys():
            fitness_sum += fitness[key]

        ids = list(range(manager.population_size))
        for key in ids:
            fitness[key] /= fitness_sum
            p.append(fitness[key])

        protected_models = manager.get_protected()
        num_protected = len(protected_models)
        num_matings = (self.ga_args.population_size - num_protected) // children_per_crossover

        mating_tuples = []
        parents = []
        for j in range(num_matings):
            mating_tuple = tuple(choice(ids, size=(mates_per_crossover,), replace=False, p=p))
            mating_tuples.append(mating_tuple)
            parents.extend(mating_tuple)

        parents = list(np.unique(parents))

        manager.update_parents(new_parents=parents)
        manager.update_mating_tuples(mating_tuples)

        return manager
