from dataclasses import dataclass, field


@dataclass
class ProblemArgs:
    """ Argument class that contains the arguments relating to problems """

    CONFIG_KEY = "problem_args"

    problem_name: str = field(default="cartpole", metadata={
        "help": "problem to solve",
    })

    batch_size: int = field(default=-1, metadata={
        "help": "batch size for training. Only relevant for classification and other dataset-based problems."
    })

    eval_batch_size: int = field(default=-1, metadata={
        "help": "batch size for inference. Only relevant for classification and other dataset-based problems."
    })

    dataset_root: str = field(default=None, metadata={
        "help": "path to the data on the local storage. Only relevant for classification and other "
                "dataset-based problems."
    })

    max_size_train: int = field(default=None, metadata={
        "help": "maximum size of the training data (in number of samples). Samples a subset randomly."
    })

    max_size_val: int = field(default=None, metadata={
        "help": "maximum size of the evaluation data (in number of samples). Samples a subset randomly."
    })

    seed: int = field(default=1337, metadata={
        "help": "seed to fix randomness"
    })

    steps: int = field(default=1, metadata={
        "help": "number of steps to take in the environment for a single run. Only relevant for rl problems."
    })

    runs: int = field(default=1, metadata={
        "help": "number of runs per evaluation. Only relevant for rl problems."
    })

    gui: bool = field(default=False, metadata={
        "help": "if the environment has a gui, display it if True. Only relevant for rl problems with a gui (OpenAI Gym"
                "problems)"
    })

    stop_on_done: bool = field(default=True, metadata={
        "help": "whether to stop taking steps until self.steps amounts of steps have elapsed regardless of whether"
                "the environment is done with the run or not"
    })
