from gaggle.problem import Problem
from gaggle.arguments import ProblemArgs, SysArgs
from gaggle.population import Individual
from gaggle.problem.environment.environment_factory import EnvironmentFactory

import torch
import numpy as np


class RLProblem(Problem):
    """Problem that uses a reinforcement learning environment as a fitness evaluation process. Used mainly
    for OpenAI gym environments but can be used for any environment that supports OpenAI Gym's environment API.

    """
    def __init__(self, problem_args: ProblemArgs = None, sys_args: SysArgs = None):
        super(RLProblem, self).__init__(problem_args, sys_args)
        self.environment = EnvironmentFactory.from_problem_args(problem_args)

    @torch.no_grad()
    def evaluate(self, individual: Individual, *args, **kwargs) -> float:
        steps = self.problem_args.steps
        runs = self.problem_args.runs
        gui = self.problem_args.gui
        stop_on_done = self.problem_args.stop_on_done

        observations = []
        rewards = []
        for r in range(runs):
            observation, _ = self.environment.reset()
            observation = torch.Tensor(observation).to(self.sys_args.device)
            run_observations = [observation]
            run_rewards = []
            for t in range(steps):
                if gui:
                    self.environment.render()
                action = individual(observation, *args, **kwargs).cpu().item()
                observation, reward, done, info, _ = self.environment.step(action)
                observation = torch.Tensor(observation).to(self.sys_args.device)
                run_observations.append(observation)
                run_rewards.append(reward)
                if stop_on_done and done:
                    break
            observations.append(run_observations)
            rewards.append(run_rewards)

        sums = [sum(run) for run in rewards]
        return np.mean(sums).item()
