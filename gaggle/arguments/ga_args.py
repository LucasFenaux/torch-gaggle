from dataclasses import dataclass, field


@dataclass
class GAArgs:
    """ Argument class that contains the arguments relating to the GA algorithms """

    CONFIG_KEY = "ga_args"

    population_size: int = field(default=100, metadata={
        "help": "number of individuals in the population to evolve."
    })

    crossover: str = field(default="uniform", metadata={
        "help": "type of crossover to use",
    })

    parent_survival_rate: float = field(default=0.5, metadata={
        "help": "probability to keep the parents rather than the children for crossover."
    })

    mutate_protected: bool = field(default=False, metadata={
        "help": "whether to mutate the protected individuals that are selected to survive (elitism)."
    })

    mutation: str = field(default="normal", metadata={
        "help": "type of mutation to use",
    })

    mutation_std: float = field(default=0.05, metadata={
        "help": "standard deviation when using normal-based random mutation"
    })

    mutation_chance: float = field(default=0.01, metadata={
        "help": "per gene probability that a gene will be mutated."
    })

    selection: str = field(default="weighted", metadata={
        "help": "type of selection to use",
    })

    elitism: float = field(default=0.1, metadata={
        "help": "% of top models (rounded down) that always get to survive to the next generation."
    })

    num_parents: int = field(default=20, metadata={
        "help": "num parents selected during the selection process"
    })

    ga_name: str = field(default="simple", metadata={
        "help": "name of the overall GA to use",
    })

    generations: int = field(default=100, metadata={
        "help": "number of generations"
    })

    k_point: int = field(default=1, metadata={
        "help": "number of points for k-point-crossover "
    })

    tournament_size: int = field(default=3, metadata={
        "help": "number of participants per tournament in tournament_selection"
    })

    selection_pressure: float = field(default=0.5, metadata={
        "help": "probability used when performing tournament selection, represents the likelihood of selecting the best"
                "performer"
    })

    uniform_mutation_min_val: float = field(default=-1., metadata={
        "help": "minimum value when sampling mutations values in uniform mutation"
    })

    uniform_mutation_max_val: float = field(default=1., metadata={
        "help": "maximum value when sampling mutations values in uniform mutation"
    })

    save_best_every: int = field(default=None, metadata={
        "help": "save best performer in the population every this many generations."
    })

    save_every_epoch: bool = field(default=False, metadata={
        "help": "force save after every epoch, independent of improvement. "
    })

    eval_every_generation: int = field(default=None, metadata={
        "help": "evaluate the population pool on the test set after this many generations"
    })

    use_freshness: bool = field(default=True, metadata={
        "help": "whether to use freshness to not recompute the fitness of surviving members that have not been modified"
                "from a generation to the next"
    })
