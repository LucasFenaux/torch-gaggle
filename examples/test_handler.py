import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


import torch

from gaggle.supervisor import GASupervisor


def fitness_function(individual):
    chromosome = individual()
    dimension = individual.get_genome_size()
    rastrigin = - (dimension * len(chromosome) + torch.sum(chromosome ** 2 - dimension * torch.cos(2 * torch.pi * chromosome)))
    return rastrigin.cpu().item()


def test():
    supervisor = GASupervisor(individual_name="pytorch", individual_size=100, device="cpu")
    supervisor.set_custom_fitness(fitness_function)
    supervisor.run()


def test_default():
    supervisor = GASupervisor(problem_name="MNIST", individual_name="nn",
                              model_name="lenet", device="cuda")
    supervisor.run()


if __name__ == '__main__':
    test_default()
