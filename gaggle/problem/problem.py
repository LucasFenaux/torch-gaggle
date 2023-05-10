from abc import ABC, abstractmethod
from gaggle.population import Individual, PopulationManager
from gaggle.arguments import ProblemArgs, SysArgs


class Problem(ABC):
    """Abstract Base Class used to define problem definitions.

        It is in charge of fitness evaluation for the problem to solve.
    """

    def __init__(self, problem_args: ProblemArgs = None, sys_args: SysArgs = None):
        super(Problem, self).__init__()
        self.problem_args = ProblemArgs() if problem_args is None else problem_args
        self.sys_args = SysArgs() if sys_args is None else sys_args

    def evaluate_population(self, population_manager: PopulationManager,
                            use_freshness: bool = True, update_manager: bool = True, train: bool = True,
                            *args, **kwargs) -> dict[int: float]:
        """
        Default behavior for evaluating the entire population, to update if it is wanted to introduce parallelization
        or other behaviors.

        Any instance of evaluate_population should always:
        - update the population fitness in the population manager by calling 
        population_manager.set_individual_fitness.
        - If freshness is being used, then it should set the population freshness in the population manager by calling
        population_manager.set_freshness.

        Args:
            population_manager:
            use_freshness:
            update_manager:

        Returns: the fitness dictionary of the population

        """
        fitness = {}
        for i in range(population_manager.population_size):
            if population_manager.is_fresh(i) and use_freshness:
                fitness[i] = None
            elif not use_freshness:
                fitness[i] = None

        for j in list(fitness.keys()):
            fitness[j] = self.evaluate(population_manager.get_individual(j), *args, **kwargs)
            if update_manager:
                population_manager.set_individual_fitness(j, fitness[j])
                if use_freshness:
                    population_manager.set_freshness(j, False)
        if train and use_freshness:
            return population_manager.get_fitness()

        return fitness

    @abstractmethod
    def evaluate(self, individual: Individual, *args, **kwargs) -> float:
        """Evaluate the given individual based on this problem.

        Args:
            individual: individual to be evaluated
            train: whether this is a training or testing evaluation (matters for certain problems)
            *args: additional args that would need to be passed to the individual forward function (call)
            **kwargs: additional kwargs that would need to be passed to the individual forward function (call)

        Returns: a float representing the individual's fitness

        """
        raise NotImplementedError
