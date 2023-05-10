import torch

from gaggle.population.individual import Individual
from gaggle.operators.crossover.crossover import Crossover
from gaggle.arguments.ga_args import GAArgs


class UniformCrossover(Crossover):
    r"""Crossover where each gene has equal probability of coming from either parent
    See the following tutorial for a more in depth description
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
    Generates two children from two parents
    """
    mates_per_crossover = 2
    children_per_crossover = 2

    def __init__(self, ga_args: GAArgs = None):
        super(UniformCrossover, self).__init__(ga_args=ga_args)

    def crossover_individual(self, individuals: list[Individual]) -> list[Individual]:
        assert len(individuals) == self.mates_per_crossover
        individual_1, individual_2 = individuals
        assert individual_1.get_genome_size() == individual_2.get_genome_size()
        assert individual_1.sys_args.device == individual_2.sys_args.device

        genome_1 = individual_1.get_gene_pool()
        genome_2 = individual_2.get_gene_pool()

        for key in genome_1.keys():
            size = genome_1[key]["param"].data.size()

            data_1 = genome_1[key]["param"].data.clone().detach()
            data_2 = genome_2[key]["param"].data.clone().detach()
            # generate the random mask
            mask_1 = torch.rand(size, dtype=torch.float, device=data_1.device) > 0.5
            mask_2 = torch.logical_not(mask_1).to(torch.float)
            mask_1 = mask_1.to(torch.float)

            genome_1[key]["param"].data = torch.mul(data_1, mask_1) + torch.mul(data_2, mask_2)
            genome_2[key]["param"].data = torch.mul(data_1, mask_2) + torch.mul(data_2, mask_1)

        return [individual_1, individual_2]


