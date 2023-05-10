import copy
from abc import abstractmethod

from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager


class Selection:
    r""" The parent class for any Selection Operator. 
    It gives functions to select the protected (elitism) set and to select the parents for crossover.

    """
    def __init__(self, ga_args: GAArgs = None):
        self.ga_args = ga_args if ga_args is not None else GAArgs()
        self.num_parents = ga_args.num_parents

    def select_protected(self, manager: PopulationManager) -> PopulationManager:
        """ By default, the select protected is elitism, to turn off elitism, set elitism to 0. in ga_args."""
        elitism = self.ga_args.elitism
        num_to_protect = int(elitism * manager.population_size)

        # we now do a topk selection process
        fitness = copy.deepcopy(manager.get_fitness())

        topk_indices = []
        for i in range(num_to_protect):
            best_idx = max(fitness, key=fitness.get)
            topk_indices.append(best_idx)
            fitness.pop(best_idx)

        manager.update_protected(new_protected=topk_indices)

        return manager

    @abstractmethod
    def select_parents(self, manager: PopulationManager, mates_per_crossover: int, children_per_crossover: int) -> PopulationManager:
        """ Should select both the parents and the mating tuples """
        raise NotImplementedError

    def select_all(self, manager: PopulationManager, mates_per_crossover: int, children_per_crossover: int) -> PopulationManager:
        """Calls both the protoected and parent selection fucntions"""
        manager = self.select_protected(manager)
        manager = self.select_parents(manager, mates_per_crossover, children_per_crossover)
        return manager
