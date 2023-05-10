import copy
from typing import Type

from gaggle.arguments.sys_args import SysArgs
from gaggle.arguments.individual_args import IndividualArgs
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.individual_factory import IndividualFactory
from gaggle.population import Individual


class PopulationManager:
    """Stores and manages all the individuals keeping track of their fitness.
     To avoid wasteful computation, we optionally keep track of an individual’s freshness and only 
     recompute its fitness if it was modified since its fitness was last computed.
     
     The Population Manager also provides a standardized interface for the evolutionary operators to access and update
     individuals. It stores all the individual-related meta-information required for the operators, such as which
     parents have been chosen, which individuals will be crossed over etc. This simplifies the operator pipeline by
     avoiding the need for operators to interface with each other directly.

    """
    def __init__(self, ga_args: GAArgs = None, individual_args: IndividualArgs = None, sys_args: SysArgs = None,
                 default_individual: Type[Individual] = None, *args, **kwargs):
        self.ga_args = ga_args if ga_args is not None else GAArgs()
        self.individual_args = individual_args if individual_args is not None else IndividualArgs()
        self.sys_args = sys_args if sys_args is not None else SysArgs()
        self.device = sys_args.device

        self.population = {}
        self.fresh = {}  # store flags to re-compute fitness, set flags to false when the model is modified
        self.fitness = {}
        self.population_size = 0

        self.protected = []  # a list of the current model indices that are protected and must survive crossover phase
        self.parents = []  # a list of the current parent indices for the current generation
        self.mating_tuples = []  # a list of the current tuples of parents to be mated in the current generation
        self.to_mutate = []  # a list of the current indices to mutate for the current generation

        self.custom_buffers = {}  # dictionary of custom buffers that can be used to store metadata for training

        for i in range(ga_args.population_size):
            if default_individual is not None:
                self.population[i] = copy.deepcopy(default_individual(individual_args, *args, **kwargs)).initialize().to(self.device)
            self.population[i] = IndividualFactory.from_individual_args(
                individual_args, self.sys_args, *args, **kwargs).initialize().to(self.device)
            self.fresh[i] = True
            self.population_size += 1

    def apply_bounds(self, ids: list[int]):
        """Applies the parameter value bounds if they were set to a value other than None in the individual_args.
        This calls each individual's own apply_bounds with the lower and upper bound.
        Args:
            ids: list of ids of the individuals to which this needs to be applied

        Returns:

        """
        if self.individual_args.param_lower_bound is not None or self.individual_args.param_upper_bound is not None:
            for individual_id in ids:
                self.population[individual_id].apply_bounds(self.individual_args.param_lower_bound,
                                                            self.individual_args.param_upper_bound)

    def create_buffer(self, key: str, initial_value=None):
        """Wrapper that allows for storage of custom buffers. This can be used to store meta-data when building custom
        operators that need to communicate with one-another.

        Args:
            key: unique key associated with the buffer that will serve as lookup
            initial_value: initial value of the buffer, can be anything

        Returns:

        """
        self.custom_buffers[key] = initial_value

    def get_buffer(self, key: str):
        """Gets a buffer associated with the key key that was created using self.create_buffer

        Args:
            key:

        Returns:

        """
        if key not in self.custom_buffers.keys():
            print(f"Buffer {key} requested does not exist")
            return None
        else:
            return self.custom_buffers[key]

    def update_buffer(self, key: str, new_value):
        """Sets the value of a buffer associated with key key that was created using self.create_buffer 
        using new value new_value.

        Args:
            key:
            new_value:

        Returns:

        """
        if key not in self.custom_buffers.keys():
            print(f"Buffer {key} requested does not exist")
            return
        else:
            self.custom_buffers[key] = new_value

    def train(self):
        """Sets the individuals to training mode. Is used if individuals have training specific configurations.

        Returns:

        """
        for i in range(self.population_size):
            self.population[i].train()

    def eval(self):
        """Sets the individuals to evaluation mode. Is used if individuals have evaluation specific configurations.

        Returns:

        """
        for i in range(self.population_size):
            self.population[i].eval()

    def is_fresh(self, individual_id: int):
        """Check if an individual with id individual_id is fresh. Meaning whether its fitness needs to 
        be recomputed or not.

        Args:
            individual_id:

        Returns:
            A boolean that reflects whether the individual's fitness needs to be recomputed or
        not. True if it does, False otherwise
        """
        return self.fresh[individual_id]

    def set_freshness(self, individual_id: int, freshness: bool):
        """Set an individual's freshness.

        Args:
            individual_id:
            freshness:

        Returns:

        """
        self.fresh[individual_id] = freshness

    def set_individual_fitness(self, individual_id: int, fitness: float):
        """Set an individual's fitness.

        Args:
            individual_id:
            fitness:

        Returns:

        """
        self.fitness[individual_id] = fitness

    def get_individual_fitness(self, individual_id: int):
        """Get an individual's fitness

        Args:
            individual_id:

        Returns:

        """
        return self.fitness[individual_id]

    def get_fitness(self):
        return self.fitness

    def get_individual(self, individual_idx: int):
        return self.population[individual_idx]

    def update_parents(self, new_parents: list[int]):
        """Update the list of parent ids with a new list of parent ids.

        Args:
            new_parents:

        Returns:

        """
        self.parents = new_parents

    def get_parents(self):
        return self.parents

    def update_mating_tuples(self, new_tuples: list[tuple]):
        """Update the list of mating tuples with a new list of mating tuples.

        Args:
            new_tuples:

        Returns:

        """
        self.mating_tuples = new_tuples

    def get_mating_tuples(self):
        return self.mating_tuples

    def update_protected(self, new_protected: list[int]):
        """Update the list of protected individual ids (individuals that will survive to the next generation no matter
        what, similar to elitism) with a new list of protected individual ids.

        Args:
            new_protected:

        Returns:

        """
        self.protected = new_protected

    def get_protected(self):
        return self.protected

    def update_to_mutate(self, new_to_mutate: list[int]):
        """Update the list of individual ids that need to be mutated during the next call to a Mutation operator with a
        new list of individual ids.

        Args:
            new_to_mutate:

        Returns:

        """
        self.to_mutate = new_to_mutate

    def get_to_mutate(self):
        return self.to_mutate

    def get_population(self):
        return self.population

    def get_freshness(self):
        return self.fresh

    def get_num_protected(self):
        return len(self.protected)

    def update_population(self, new_individuals: dict[int: Individual], new_freshness: dict[int: bool]):
        """Update the entire population as well as its freshness with new dictionaries of {id: Individual} and
        {id: bool}.

        Args:
            new_individuals:
            new_freshness:

        Returns:

        """
        self.population = new_individuals
        self.fresh = new_freshness

    def get_gene_count(self):
        """Assuming all individuals have the same genome size, returns the genome size of the first individual (which
        if the assumption holds should be the genome size of all individuals in the population.

        Returns:
            Genome size as an int.
        """
        return self.population[0].get_genome_size()
