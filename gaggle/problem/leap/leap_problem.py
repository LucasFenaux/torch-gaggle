from gaggle.problem.problem import Problem
from gaggle.arguments import ProblemArgs, SysArgs
from typing import Callable
from gaggle.population import Individual


def convert_leap_problem(leap_problem: Callable, *args, **kwargs):
    return RegistrableLeapProblem(leap_problem, *args, **kwargs)


class RegistrableLeapProblem:
    """Constructor that returns its associated leap problem when called (wrapper for ProblemFactory).

    """
    def __init__(self,  leap_problem: Callable, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.leap_problem = leap_problem

    def __call__(self, problem_args: ProblemArgs = None, sys_args: SysArgs = None, *args, **kwargs):
        return LeapProblem(self.leap_problem, problem_args=problem_args, sys_args=sys_args, *self.args,
                           **self.kwargs)


class LeapProblem(Problem):
    """Wrapper for leap problems. Takes in a leap problem and allows it to be run within Gaggle's framework.

    """
    def __init__(self, leap_problem: Callable, problem_args: ProblemArgs = None, sys_args: SysArgs = None,
                 *args, **kwargs):
        super(LeapProblem, self).__init__(problem_args, sys_args)
        self.leap_problem = leap_problem(*args, **kwargs)

    def evaluate(self, individual: Individual, *args, **kwargs) -> float:
        params = individual(*args, **kwargs)
        return self.leap_problem.evaluate(params)
