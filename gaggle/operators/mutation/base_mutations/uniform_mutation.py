import torch

from gaggle.population.base_individuals.nn_individual import Individual
from gaggle.operators.mutation.mutation import Mutation
from gaggle.arguments.ga_args import GAArgs


class UniformMutation(Mutation):
    r"""For real valued chromosomes
    Adds noise from a Uniform distribution within the range specified by:
    ga_args.uniform_mutation_min_val and ga_args.uniform_mutation_max_val
    Noise is only added to each gene with probability specified by ga_args.mutation_chance
    """
    def __init__(self, ga_args: GAArgs = None):
        super(UniformMutation, self).__init__(ga_args)
        self.uniform_mutation_min_val = self.ga_args.uniform_mutation_min_val
        self.uniform_mutation_max_val = self.ga_args.uniform_mutation_max_val

    def mutate_individual(self, individual: Individual) -> Individual:
        genome = individual.get_gene_pool()
        num_chromosomes = len(genome.keys())
        for i in range(num_chromosomes):
            # generate the random mask
            mask = torch.rand(genome[i]["param"].data.size(), dtype=torch.float,
                                device=genome[i]["param"].data.device) < self.ga_args.mutation_chance
            indices = torch.nonzero(mask, as_tuple=True)
            scaled_mutation = (torch.rand(size=genome[i]["param"].data[indices].size(), device=genome[i]["param"].data.device) *
                       (self.uniform_mutation_max_val - self.uniform_mutation_min_val)) - self.uniform_mutation_min_val
            genome[i]["param"].data[indices] += scaled_mutation
        return individual
