
from gaggle.population.individual import Individual
from gaggle.operators.crossover.crossover import Crossover
from gaggle.arguments.ga_args import GAArgs
from gaggle.utils.individual_helper import from_gene_pool, from_tensor

import torch
from numpy.random import default_rng
import numpy as np


class KPointCrossover(Crossover):
    r"""Generalization of single point crossover to k points
    See the following tutorial for a more in depth description
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
    Generates two children from two parents
    """
    mates_per_crossover = 2
    children_per_crossover = 2

    def __init__(self, ga_args: GAArgs = None):
        super(KPointCrossover, self).__init__(ga_args=ga_args)

    def crossover_individual(self, individuals: list[Individual]) ->list[Individual]:
        assert len(individuals) == self.mates_per_crossover
        individual_1, individual_2 = individuals
        assert individual_1.get_genome_size() == individual_2.get_genome_size()
        assert individual_1.sys_args.device == individual_2.sys_args.device
        genome_size = individual_1.get_genome_size()

        genome_1 = individual_1.get_gene_pool()
        genome_2 = individual_2.get_gene_pool()

        # convert to a singular tensor
        tensor_1, metadata_1 = from_gene_pool(genome_1)
        tensor_2, metadata_2 = from_gene_pool(genome_2)

        k = self.ga_args.k_point

        # select a set of k random indices without repeats
        rng = default_rng()
        cut_indices = np.sort(rng.choice(genome_size+1, size=k, replace=False))
        flip = True
        last_cut = 0
        for cut_idx in cut_indices:
            if flip:
                data_1 = tensor_1[last_cut:cut_idx].clone().detach()
                tensor_1[last_cut: cut_idx] = tensor_2[last_cut:cut_idx].clone().detach()
                tensor_2[last_cut: cut_idx] = data_1

            last_cut = cut_idx
            flip = not flip

        # don't forget the last one
        if flip:
            data_1 = tensor_1[last_cut:].clone().detach()
            tensor_1[last_cut:] = tensor_2[last_cut:].clone().detach()
            tensor_2[last_cut:] = data_1

        from_tensor(gene_pool=genome_1, tensor=tensor_1, metadata=metadata_1)
        from_tensor(gene_pool=genome_2, tensor=tensor_2, metadata=metadata_2)

        return [individual_1, individual_2]

    def deprecated_crossover_individual(self, individuals: list[Individual]) -> list[Individual]:
        # TODO: finish implementing, might need a refactor to make it smarter/easier to read
        assert len(individuals) == self.mates_per_crossover
        individual_1, individual_2 = individuals
        assert individual_1.get_genome_size() == individual_2.get_genome_size()
        assert individual_1.sys_args.device == individual_2.sys_args.device
        genome_size = individual_1.get_genome_size()

        genome_1 = individual_1.get_gene_pool()
        genome_2 = individual_2.get_gene_pool()

        k = self.ga_args.k_point

        # select a set of k random indices without repeats
        rng = default_rng()
        cut_indices = np.sort(rng.choice(genome_size+1, size=k, replace=False))
        flip = True

        count = 0
        curr = genome_1[count]["gene_size"]
        last_cut = 0
        for i, cut_idx in enumerate(cut_indices):
            while curr < cut_idx-1:
                count += 1
                curr += genome_1[count]["gene_size"]

                if curr < cut_idx-1:
                    if flip:
                        data_1 = genome_1[count]["param"].data.clone().detach()
                        data_2 = genome_2[count]["param"].data.clone().detach()
                        genome_1[count]["param"].data = data_2
                        genome_2[count]["param"].data = data_1

            # we reached the param where the cut is located
            if flip:
                end_cut = cut_idx - (curr - genome_1[count]["gene_size"])
                start_cut = max(0, last_cut - (curr - genome_1[count]["gene_size"]))

                size_1 = genome_1[count]["param"].data.size()
                size_2 = genome_2[count]["param"].data.size()
                data_1 = genome_1[count]["param"].data.clone().detach().flatten()
                data_1_copy = data_1.clone().detach()
                data_2 = genome_2[count]["param"].data.clone().detach().flatten()

                data_1[start_cut:end_cut] = data_2[start_cut:end_cut]
                data_2[start_cut:end_cut] = data_1_copy[start_cut:end_cut]

                data_1 = torch.unflatten(data_1, 0, size_1)
                data_2 = torch.unflatten(data_2, 0, size_2)

                genome_1[count]["param"].data = data_1
                genome_2[count]["param"].data = data_2

            last_cut = cut_idx
            flip = not flip

        # we do the last swap for the last part (otherwise 2 point is actually one point crossover
        if flip:
            start_cut = max(0, last_cut - (curr - genome_1[count]["gene_size"]))

            size_1 = genome_1[count]["param"].data.size()
            size_2 = genome_2[count]["param"].data.size()
            data_1 = genome_1[count]["param"].data.clone().detach().flatten()
            data_1_copy = data_1.clone().detach()
            data_2 = genome_2[count]["param"].data.clone().detach().flatten()

            data_1[start_cut:] = data_2[start_cut:]
            data_2[start_cut:] = data_1_copy[start_cut:]

            data_1 = torch.unflatten(data_1, 0, size_1)
            data_2 = torch.unflatten(data_2, 0, size_2)

            genome_1[count]["param"].data = data_1
            genome_2[count]["param"].data = data_2

        return [individual_1, individual_2]


