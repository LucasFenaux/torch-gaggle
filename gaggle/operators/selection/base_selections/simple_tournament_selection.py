import copy
from numpy.random import choice
import numpy as np
from gaggle.operators.selection.selection import Selection
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager


class SimpleTournamentSelection(Selection):
    r"""Standard Tournament selection where we uniformly choose k Individuals
    where k is specified by ga_args.tournament_size
    Then we simply return the best out of the set of k.
    This is repeated for the total number of parents needed IID.
    
    """
    def __init__(self, ga_args: GAArgs = None):
        super(SimpleTournamentSelection, self).__init__(ga_args=ga_args)

    def tournament(self, participant_ids: np.array, participants_fitness: np.array):
        assert len(participant_ids) == self.ga_args.tournament_size
        assert len(participant_ids) == len(participants_fitness)
        max_index = np.argmax(participants_fitness)
        winner = participant_ids[max_index]
        return winner

    def select_parents(self, manager: PopulationManager, mates_per_crossover: int, children_per_crossover: int) -> PopulationManager:
        assert 0. < self.ga_args.selection_pressure <= 1.
        fitness = copy.deepcopy(manager.get_fitness())
        fitness = np.array(list(fitness.values()))
        ids = np.array(list(range(manager.population_size)))

        protected_models = manager.get_protected()
        num_protected = len(protected_models)
        num_matings = (self.ga_args.population_size - num_protected) // children_per_crossover

        mating_tuples = []
        parents = []
        for i in range(num_matings):
            mating_list = []
            for j in range(mates_per_crossover):
                participant_ids = choice(ids, size=self.ga_args.tournament_size)
                participants_fitness = fitness[participant_ids]
                mating_list.append(self.tournament(participant_ids, participants_fitness))

            mating_tuples.append(tuple(mating_list))
            parents.extend(tuple(mating_list))

        parents = list(np.unique(parents))
        manager.update_parents(new_parents=parents)
        manager.update_mating_tuples(mating_tuples)

        return manager
