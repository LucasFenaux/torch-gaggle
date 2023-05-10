from gaggle.problem import Problem
from gaggle.arguments import ProblemArgs, SysArgs
from gaggle.population import Individual

import torch
from typing import Callable


class FunctionalProblem(Problem):
    """Problem that just uses a predefined fitness_function with default input arguments beyond the individual.

    """
    def __init__(self, fitness_function: Callable, problem_args: ProblemArgs = None, sys_args: SysArgs = None, *args,
                 **kwargs):
        super(FunctionalProblem, self).__init__(problem_args, sys_args)
        self.fitness_function = fitness_function
        self.args = args
        self.kwargs = kwargs

    @torch.no_grad()
    def evaluate(self, individual: Individual, *args, **kwargs) -> float:
        # we don't use the provided *args, **kwargs as this is intended for the supervisor and the user would not
        # have direct access to this part of the pipeline and therefore the *args and **kwargs are set at init time
        return self.fitness_function(individual, *self.args, **self.kwargs)
