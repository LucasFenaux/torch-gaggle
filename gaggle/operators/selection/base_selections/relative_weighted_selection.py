import copy
from numpy.random import choice
import numpy as np
from gaggle.operators.selection.selection import Selection
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager


class RelativeWeightedSelection(Selection):
    r"""A variation on Roulette wheel that subtracts the worst fittness from each candidate
    Note the worst fittness is *0.99 so that the candidate with the worst fittness is not assigned 0% chance
    
    """
    def __init__(self, ga_args: GAArgs = None):
        super(RelativeWeightedSelection, self).__init__(ga_args=ga_args)

    def select_parents(self, manager: PopulationManager, mates_per_crossover: int, children_per_crossover: int) -> PopulationManager:
        fitness = copy.deepcopy(manager.get_fitness())
        ids = list(range(manager.population_size))

        worst_idx = min(fitness, key=fitness.get)
        worst_fitness = 0.99*fitness[worst_idx]  # 0.99 to still give the worst performer a chance to survive (small)

        # we scale by the poorest performer so that the best performer get attention proportional to their relative performance
        fitness_sum = 0.
        for key in fitness.keys():
            fitness[key] -= worst_fitness
            fitness_sum += fitness[key]

        p = []

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
