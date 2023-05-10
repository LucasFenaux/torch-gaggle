import torch

from gaggle.population.base_individuals.nn_individual import Individual
from gaggle.operators.mutation.mutation import Mutation
from gaggle.arguments.ga_args import GAArgs


class NormalMutation(Mutation):
    r"""For real valued chromosomes
    Adds noise from a Gaussian distribution with standard deviation ga_args.mutation_std
    Noise is only added to each gene with probability specified by ga_args.mutation_chance
    """
    def __init__(self, ga_args: GAArgs = None):
        super(NormalMutation, self).__init__(ga_args=ga_args)

    def mutate_individual(self, individual: Individual) -> Individual:
        genome = individual.get_gene_pool()
        num_chromosomes = len(genome.keys())
        for i in range(num_chromosomes):
            # generate the random mask
            mask = torch.rand(genome[i]["param"].data.size(), dtype=torch.float,
                                device=genome[i]["param"].data.device) < self.ga_args.mutation_chance
            indices = torch.nonzero(mask, as_tuple=True)
            genome[i]["param"].data[indices] += torch.normal(mean=0., std=self.ga_args.mutation_std,
                                                             size=genome[i]["param"].data[indices].size(),
                                                             device=genome[i]["param"].data.device)
        return individual
