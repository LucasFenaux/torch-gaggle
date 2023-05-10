from gaggle.arguments import ConfigArgs, ProblemArgs, SysArgs, GAArgs, IndividualArgs, OutdirArgs
from gaggle.population import PopulationManager
from gaggle.ga import GAFactory, GA
from gaggle.utils.special_print import print_warning
from typing import Callable
from gaggle.problem import FunctionalProblem


class GASupervisor:
    """Gives a single-line interface to our framework. At minimum,
       a user needs to specify which problem they want to solve.
       Also allows for customization by passing functions or new operators.

    """
    def __init__(self, ga_name: str = "simple", population_size: int = 100, num_parents: int = 100,
                 generations: int = 100, problem_name: str = "MNIST", individual_name: str = "nn",
                 individual_size: int = 100, device: str = "cpu", elitism: float = 0.1, crossover: str = "uniform",
                 k_point: int = 1, tournament_size: int = 3, selection_pressure: float = 0.5, mutation: str = "normal",
                 mutation_chance: float = 0.01, mutation_std: float = 0.05, use_freshness: bool = True,
                 mutate_protected: bool = False, uniform_mutation_min_val: float = -1.,
                 uniform_mutation_max_val: float = 1., selection: str = "weighted", parent_survival_rate: float = 0.5,
                 batch_size: int = -1, eval_batch_size: int = -1, save_best_every: int = None,
                 eval_every_generation: int = 50, model_name: str = "lenet", root: str = "./runs", name: str = "run",
                 seed: int = 1337, steps: int = 1, runs: int = 1, dataset_root: str = "None", gui: bool = False,
                 display_train_metrics: bool = True, display_test_metrics: bool = True):
        """Initialize the GASupervisor..

        Args:
            ga_name: name of the overall GA to use.
            population_size: number of individuals in the population to evolve.
            num_parents: num parents selected during the selection process (recommended value is = population_size).
            generations: number of generations.
            problem_name: name of the problem, if custom, set_custom_fitness needs to be called to setup the custom
            fitness formula.
            individual_name: the type of individual to use to represent the solutions to evolve.
            individual_size: length of the parameter tensor for the basic NumpyIndividual and PytorchIndividual.
            This argument is irrelevant for other individuals (unless is has been customized).
            device: device to run algorithms on. Can be a torch.device object or a str ("cpu" or "cuda").
            elitism: % of top models (rounded down) that always gets to survive to the next generation.
            crossover: type of crossover to use.
            k_point: number of points for k-point-crossover.
            tournament_size: number of participants per tournament in tournament_selection.
            selection_pressure: probability used when performing tournament selection, represents the likelihood of
            selecting the best performer.
            mutation: type of mutation to use.
            mutation_chance: per gene probability that a gene will be mutated.
            mutation_std: standard deviation when using normal-based random mutation.
            use_freshness: whether to use freshness to not recompute the fitness of surviving members that have not
            been modified from a generation to the next.
            mutate_protected: whether to mutate the protected individuals that are selected to survive (elitism).
            uniform_mutation_min_val: minimum value when sampling mutations values in uniform mutation.
            uniform_mutation_max_val: maximum value when sampling mutations values in uniform mutation.
            selection: type of selection to use.
            parent_survival_rate: (aka probability of crossover) probability to keep the parents rather than the children for crossover.
            batch_size: batch size for training. Only relevant for classification and other dataset-based problems.
            eval_batch_size: batch size for inference. Only relevant for classification and other dataset-based
            problems.
            save_best_every: save best performer in the population every this many generations.
            eval_every_generation: evaluate the population pool on the test set after this many generations.
            model_name: name of the model architecture. Only relevant for neural network individuals.
            root: Root folder where to put the experiments (good choice can be f'{os.getcwd()}/experiments').
            name: Name of each experiment folder.
            seed: seed to fix randomness.
            steps: number of steps to take in the environment for a single run. Only relevant for rl problems.
            runs: number of runs per evaluation. Only relevant for rl problems.
            dataset_root: path to the data on the local storage. Only relevant for classification and other
            dataset-based problems.
            gui: if the environment has a gui, display it if True. Only relevant for rl problems with a gui (OpenAI Gym problems).
            display_train_metrics: whether to draw a graph with the train metrics at the end of training, needs at
             least 11 generations of training as the default window size is 10.
            display_test_metrics: whether to draw a graph with the train metrics at the end of training, needs
            number of generation / eval_every_generation > 10 to draw anything since the default window size is 10.
        """
        self.config_args = ConfigArgs()
        self.problem_args = ProblemArgs()
        self.sys_args = SysArgs()
        self.ga_args = GAArgs()
        self.individual_args = IndividualArgs()
        self.outdir_args = OutdirArgs()
        # replace the init arguments
        self.ga_args.ga_name = ga_name
        self.ga_args.population_size = population_size
        self.ga_args.num_parents = num_parents
        self.ga_args.generations = generations
        self.problem_args.problem_name = problem_name
        self.problem_args.dataset_root = dataset_root
        self.problem_args.seed = seed
        self.problem_args.steps = steps
        self.problem_args.runs = runs
        self.problem_args.gui = gui
        self.sys_args.device = device
        self.ga_args.crossover = crossover
        self.ga_args.k_point = k_point
        self.ga_args.tournament_size = tournament_size
        self.ga_args.selection_pressure = selection_pressure
        self.ga_args.elitism = elitism
        self.ga_args.mutation = mutation
        self.ga_args.mutate_protected = mutate_protected
        self.ga_args.mutation_chance = mutation_chance
        self.ga_args.mutation_std = mutation_std
        self.ga_args.uniform_mutation_min_val = uniform_mutation_min_val
        self.ga_args.uniform_mutation_max_val = uniform_mutation_max_val
        self.ga_args.selection = selection
        self.ga_args.parent_survival_rate = parent_survival_rate
        self.problem_args.batch_size = batch_size
        self.problem_args.eval_batch_size = eval_batch_size
        self.ga_args.save_best_every = save_best_every
        self.ga_args.eval_every_generation = eval_every_generation
        self.ga_args.use_freshness = use_freshness
        self.individual_args.model_name = model_name
        self.individual_args.individual_name = individual_name
        self.individual_args.individual_size = individual_size
        self.outdir_args.root = root
        self.outdir_args.name = name
        self.custom_fitness_function = None
        self.display_train_metrics = display_train_metrics
        self.display_test_metrics = display_test_metrics
        self.args = []
        self.kwargs = {}

    def set_custom_fitness(self, fitness_function: Callable, *args, **kwargs):
        """If during initialization, problem_name is set to "custom", then this function needs to be called to
        setup the fitness function to evaluate the population on before calling self.run(). This takes in any callable
        that will return a float value of the individual. If custom arguments are necessary, they can be passed
        as *args and **kwargs and will be used when invoking the fitness_function.

        Args:
            fitness_function: fitness function to optimize
            *args:
            **kwargs:

        Returns:

        """
        self.custom_fitness_function = fitness_function
        self.args = args
        self.kwargs = kwargs

    def _run_default(self, *args, **kwargs):
        """Run script for default parameters and default fitness function.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        population_manager: PopulationManager = PopulationManager(self.ga_args, self.individual_args,
                                                                  sys_args=self.sys_args, *args, **kwargs)
        trainer: GA = GAFactory.from_ga_args(population_manager=population_manager, ga_args=self.ga_args,
                                             problem_args=self.problem_args, sys_args=self.sys_args,
                                             outdir_args=self.outdir_args, individual_args=self.individual_args)
        trainer.train(display_train_metrics=self.display_train_metrics, display_test_metrics=self.display_test_metrics)

    def _run_custom(self, *args, **kwargs):
        """Run script for a custom fitness function.

        Returns:

        """
        if self.custom_fitness_function is None:
            print_warning(f"Attempted to run a custom fitness function but the fitness function was not set")
            return

        # we first initialize the population
        population_manager: PopulationManager = PopulationManager(ga_args=self.ga_args,
                                                                  individual_args=self.individual_args,
                                                                  sys_args=self.sys_args, *args, **kwargs)
        # we then define the problem
        problem = FunctionalProblem(fitness_function=self.custom_fitness_function, problem_args=self.problem_args,
                                    sys_args=self.sys_args, *self.args, **self.kwargs)
        trainer: GA = GAFactory.from_ga_args(population_manager=population_manager, problem=problem,
                                             ga_args=self.ga_args,
                                             problem_args=self.problem_args, sys_args=self.sys_args,
                                             outdir_args=self.outdir_args, individual_args=self.individual_args)
        trainer.train(display_train_metrics=self.display_train_metrics, display_test_metrics=self.display_test_metrics)

    def run(self, *args, **kwargs):
        """Run the genetic algorithm described during __init__. When running pre-existing problems, if additional
        parameters need to be passed to the initialization of the PopulationManager, they can be given in *args and
        **kwargs.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        if self.problem_args.problem_name != "custom":
            self._run_default(*args, **kwargs)
        else:
            self._run_custom(*args, **kwargs)
