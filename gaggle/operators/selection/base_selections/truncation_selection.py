import copy
from gaggle.operators.selection.selection import Selection
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager
import torch


class TruncationSelection(Selection):
    r"""Selects the best parents determinisitcially 
    e.g. if we want 10 parents, we choose the individuals with the top-10 fittness
    """
    def __init__(self, ga_args: GAArgs = None):
        super(TruncationSelection, self).__init__(ga_args=ga_args)

    def select_parents(self, manager: PopulationManager, mates_per_crossover: int, children_per_crossover: int) -> PopulationManager:
        fitness = copy.deepcopy(manager.get_fitness())

        parents = []
        for i in range(self.num_parents):
            best_idx = max(fitness, key=fitness.get)
            parents.append(best_idx)
            fitness.pop(best_idx)

        manager.update_parents(new_parents=parents)

        # first we get the number of protected
        num_protected = manager.get_num_protected()
        num_matings = (self.ga_args.population_size - num_protected) // children_per_crossover

        num_parents = len(parents)

        # generating the mating tuples
        mating_tuples = []
        for j in range(num_matings):
            # rand = torch.randint(low=0, high=num_parents, size=(self.mates_per_crossover,))
            # using randperm to avoid duplicates
            rand = torch.randperm(num_parents)[:mates_per_crossover]
            mating_tuples.append(tuple(parents[rand[i]] for i in range(rand.size(0))))

        manager.update_mating_tuples(mating_tuples)

        return manager
