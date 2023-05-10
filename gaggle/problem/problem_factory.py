from gaggle.problem import Problem
from gaggle.arguments import ProblemArgs, SysArgs
from gaggle.problem.environment.environment_factory import EnvironmentFactory
from gaggle.problem.environment.rl_problem import RLProblem
from gaggle.problem.leap.leap_problem import convert_leap_problem
from gaggle.problem.dataset.classification_problem import ClassificationProblem
from gaggle.problem.dataset.dataset_factory import DatasetFactory
from gaggle.utils.special_print import print_warning
from typing import Callable


class ProblemFactory:
    r"""Factory that generates pre-existing available Problems.
    ProblemFactory.problems stores said Problems as a dictionary. It stores problems as 4 types: classification,
    rl, leap and custom. Each is a key that holds a dictionary of available problems/datasets/environments.
    """
    problems = {
        "classification": DatasetFactory.datasets,
        "rl": EnvironmentFactory.environments,
        "leap": {},
        "custom": {}
    }

    registrable_problems = ["leap", "custom"]

    @classmethod
    def register_problem(cls, problem_type: str, problem_name: str, problem: Callable, *args, **kwargs):
        """Register a new problem to the factory of pre-existing available Problems. Said problem has to be a problem
        that is registrable. ProblemFactory.registrable_problems stores the list of problem types that are allowed to
        be registrables. For the other types, please check their respective documentation to see how to register them
        (for classification: ClassificationProblem and DatasetFactory, for rl: RLProblem and EnvironmentFactory).
        
        If custom arguments need to be passed at initialization time of the Problem (for example when
        ProblemFactory.from_problem_args is called), they can also be passed as *args and **kwargs and they will be
        used to initialize the problem.
        Args:
            problem_type: The key to ProblemFactory.problems.
            problem_name: The new problem's name as a str.
            problem: The uninitialized problem.
            *args: The args that need to be passed to the problem at initialization time
            **kwargs: the kwargs that need to be passed to the problem at initialization time.

        Returns:

        """
        assert isinstance(problem, Callable)
        if problem_type not in cls.registrable_problems:
            if problem_type == "classification":
                print_warning(f"To register a new classification problem that has the same evaluate as "
                              f"ClassificationProblem but a different dataset,"
                              f" please use DatasetFactory.update with the updated dataset.\n"
                              f"If the behavior is modified, then register the new problem as 'custom'")
            elif problem_type == "rl":
                print_warning(f"To register a new rl problem that has the same evaluate as RLProblem but a different "
                              f"environment, please use EnvironmentFactory.update with the updated environment.\n"
                              f"If the behavior is modified, then register the new problem as 'custom'")
            else:
                print(f"Can only register new {cls.registrable_problems} problems")
                raise ValueError(problem_type)
        else:
            if problem_name in list(cls.problems[problem_type].keys()):
                print_warning(f"{problem_name} is an already existing {problem_type} problem and will be overwritten")

            cls.problems[problem_type][problem_name] = (problem, args, kwargs)

    @classmethod
    def from_problem_args(cls, problem_args: ProblemArgs = None, sys_args: SysArgs = None) -> Problem:
        r"""Initializes the requested Problem from the dictionary of available Problems.

        This is done by using the attributes problem_args.problem_type and problem_args.problem_name as
        the lookup keys to ProblemFactory.problems.

        Args:
            problem_args:
            sys_args:

        Returns:
            The initialized Problem requested.
        """
        problem_args = problem_args if problem_args is not None else ProblemArgs()
        sys_args = SysArgs() if sys_args is None else sys_args

        problem_type = None
        for key, value in cls.problems.items():
            if problem_args.problem_name in list(value.keys()):
                problem_type = key
                break
        if problem_type is None:
            raise ValueError(problem_args.problem_name)
        if problem_type == "classification":
            return ClassificationProblem(problem_args, sys_args)
        elif problem_type == "rl":
            return RLProblem(problem_args, sys_args)
        else:
            problem, args, kwargs = cls.problems[problem_type][problem_args.problem_name]
            return problem(problem_args, sys_args, *args, **kwargs)

    @classmethod
    def convert_and_register_leap_problem(cls, problem_name: str, leap_problem: Callable, *args, **kwargs):
        """Shortcut method that both converts the leap problem to our Problem class and registers it

        Args:
            problem_name:
            leap_problem:
            *args:
            **kwargs:

        Returns:

        """
        registrable_leap_problem = convert_leap_problem(leap_problem, *args, **kwargs)
        cls.register_problem(problem_type="leap", problem_name=problem_name, problem=registrable_leap_problem, *args,
                             **kwargs)
