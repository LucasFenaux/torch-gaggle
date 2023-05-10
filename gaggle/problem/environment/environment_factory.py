from typing import Callable
import gym
from gaggle.arguments import ProblemArgs


class GymWrapper:
    """ Callable wrapper that returns the gym environment"""
    def __init__(self, environment_name: str):
        self.environment_name = environment_name

    def __call__(self, *args, **kwargs):
        return gym.make(self.environment_name)


class EnvironmentFactory:
    r"""Factory that generates available environments.
    EnvironmentFactory.environments stores said environments as a dictionary with their name as key and the
    uninitialized Environment object as value.

    See Also:
        Environment Class
    """

    environments = {
        "cartpole": GymWrapper("CartPole-v1"),
    }

    @classmethod
    def get_keys(cls):
        r"""Gets the keys (environment names) for the available pre-built environment.

        Returns:
            list of strings that are the keys to EnvironmentFactory.environments

        """
        return list(cls.environments.keys())

    @classmethod
    def update(cls, key, environment):
        r"""Add a new dataset to the dictionary of datasets that can be created.

        It is added to EnvironmentFactory.environments

        Args:
            key: dataset name that will be used as the dictionary lookup key
            environment: environment class object, it needs to not be already initialized

        """
        assert isinstance(environment, Callable)
        cls.environments[key] = environment

    @classmethod
    def from_problem_args(cls, problem_args: ProblemArgs):
        environment = cls.environments.get(problem_args.problem_name, None)
        if environment is None:
            raise ValueError(problem_args.problem_name)

        return environment(problem_args)

    @classmethod
    def from_gym_env_id(cls, env_id: str):
        """
        Takes in a gym env id and returns the associated OpenAI Gym Environment
        Args:
            env_id: gym environment id

        Returns: Gym environment

        """

        return gym.make(env_id)
