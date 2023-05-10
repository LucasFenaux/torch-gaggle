from abc import abstractmethod


from gaggle.arguments import GAArgs, SysArgs, ProblemArgs, OutdirArgs, IndividualArgs
from gaggle.operators import Crossover, Mutation, Selection, CrossoverFactory, MutationFactory, SelectionFactory
from gaggle.problem import Problem, ProblemFactory
from gaggle.population import PopulationManager


class GA:
    """ The parent class for any GA. Is used to store all the information related to the GA algorithm and
    organize the order of the operators.

    """
    def __init__(self, population_manager: PopulationManager = None, ga_args: GAArgs = None, selection: Selection = None,
                 crossover: Crossover = None, mutation: Mutation = None, problem_args: ProblemArgs = None,
                 sys_args: SysArgs = None, outdir_args: OutdirArgs = None, individual_args: IndividualArgs = None,
                 problem: Problem = None):
        self.sys_args = sys_args if sys_args is not None else SysArgs()
        self.ga_args = ga_args if ga_args is not None else GAArgs()
        self.outdir_args = outdir_args if outdir_args is not None else OutdirArgs()
        self.problem_args = problem_args if problem_args is not None else ProblemArgs()
        self.individual_args = individual_args if individual_args is not None else IndividualArgs()

        self.problem: Problem = problem if problem is not None else ProblemFactory.from_problem_args(problem_args,
                                                                                                     sys_args=sys_args)

        self.population_manager = population_manager if population_manager is not None else PopulationManager(
            ga_args=self.ga_args, individual_args=self.individual_args, sys_args=self.sys_args)

        if selection is None:
            self.selection_fn = SelectionFactory.from_ga_args(ga_args)
        else:
            self.selection_fn = selection
        if crossover is None:
            self.crossover_fn = CrossoverFactory.from_ga_args(ga_args)
        else:
            self.crossover_fn = crossover
        if mutation is None:
            self.mutation_fn = MutationFactory.from_ga_args(ga_args)
        else:
            self.mutation_fn = mutation

    @abstractmethod
    def train(self, *args, **kwargs):
        """ Function used to evolve the GA. Needs to be overwritten.
        """
        raise NotImplementedError
