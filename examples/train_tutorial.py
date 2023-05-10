import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataclasses import dataclass, field
from typing import Union
import copy

from gaggle.arguments import ConfigArgs
from gaggle.arguments.problem_args import ProblemArgs
from gaggle.arguments.sys_args import SysArgs
from gaggle.arguments.individual_args import IndividualArgs
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.arguments.ga_args import GAArgs
from gaggle.population.population_manager import PopulationManager
from gaggle.utils.special_print import print_dict_highlighted
from gaggle.ga import GA, SimpleGA
from gaggle.ga.ga_factory import GAFactory
from gaggle.operators import CrossoverFactory, Crossover, MutationFactory, Mutation, SelectionFactory, Selection
from gaggle.population import Individual, SupervisedIndividual, IndividualFactory

import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class MyGAArgs(GAArgs):

    # can set new fields
    new_stuff: int = field(default=0, metadata={
        "help": "new stuff"
    })

    # can overwrite existing fields. Do at your own risks, not recommended
    population_size: int = field(default=100, metadata={
        "help": "testing stuff"
    })


ConfigArgs.update(GAArgs.CONFIG_KEY, MyGAArgs)


class MyCrossover(Crossover):
    children_per_crossover = 2
    mates_per_crossover = 2

    def __init__(self, ga_args: MyGAArgs):
        super(MyCrossover, self).__init__(ga_args=ga_args)
        print("Works!")

    def crossover_individual(self, individuals: list[Individual]) -> list[Individual]:
        assert len(individuals) == self.mates_per_crossover
        # model_1, model_2 = copy.deepcopy(models)
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


CrossoverFactory.update("custom", MyCrossover)


class MyMutation(Mutation):
    def __init__(self, ga_args: MyGAArgs):
        super(MyMutation, self).__init__(ga_args=ga_args)

    def mutate_individual(self, individual: Individual) -> Individual:
        return individual


MutationFactory.update("custom", MyMutation)


class MySelection(Selection):
    def __init__(self, ga_args: MyGAArgs):
        super(MySelection, self).__init__(ga_args=ga_args)

    def select_parents(self, manager: PopulationManager, mates_per_crossover: int,
                       children_per_crossover: int) -> PopulationManager:
        fitness = copy.deepcopy(manager.get_fitness())

        parents = []
        for i in range(self.num_parents):
            best_idx = max(fitness, key=fitness.get)
            parents.append(best_idx)
            fitness.pop(best_idx)

        manager.update_parents(new_parents=parents)

        # first we get the number of protected
        num_protected = manager.get_num_protected()
        num_matings = (self.ga_args.population_size - num_protected) // children_per_crossover

        num_parents = len(parents)

        # generating the mating tuples
        mating_tuples = []
        for j in range(num_matings):
            # rand = torch.randint(low=0, high=num_parents, size=(self.mates_per_crossover,))
            # using randperm to avoid duplicates
            rand = torch.randperm(num_parents)[:mates_per_crossover]
            mating_tuples.append(tuple(parents[rand[i]] for i in range(rand.size(0))))

        manager.update_mating_tuples(mating_tuples)

        return manager


SelectionFactory.update("custom", MySelection)


class MyGA(SimpleGA):
    def __init__(self, ga_args: GAArgs, selection: Selection = None,
                 crossover: Crossover = None, mutation: Mutation = None, sys_args: SysArgs = None):
        super(MyGA, self).__init__(ga_args, selection, crossover, mutation, sys_args)

    def train_one_generation(self):
        """
        Standard one generation GA pipeline
        """
        self.population_manager.train()
        train_fitness = self.problem.evaluate_population(self.population_manager,
                                                         use_freshness=self.ga_args.use_freshness, update_manager=True,
                                                         train=True)
        print("Got fitness")
        population_manager = self.selection_fn.select_all(self.population_manager,
                                                          self.crossover_fn.mates_per_crossover,
                                                          self.crossover_fn.children_per_crossover)
        population_manager = self.crossover_fn.crossover_pop(population_manager)
        population_manager = self.mutation_fn.mutate_pop(population_manager)
        return train_fitness, population_manager


GAFactory.update("custom", MyGA)


class MyIndividual(SupervisedIndividual):
    def __init__(self, individual_args: IndividualArgs, sys_args: SysArgs = None, model: nn.Module = None):
        super(MyIndividual, self).__init__(individual_args, sys_args, model)

    def evaluate(self, data: Union[tuple[torch.Tensor, torch.Tensor], DataLoader]) -> float:
        print("test")
        return 0.


IndividualFactory.update("custom", MyIndividual)


def parse_args():
    parser = transformers.HfArgumentParser((OutdirArgs, SysArgs, IndividualArgs, MyGAArgs, ProblemArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train(outdir_args: OutdirArgs,
          sys_args: SysArgs,
          individual_args: IndividualArgs,
          ga_args: MyGAArgs,
          problem_args: ProblemArgs,
          config_args: ConfigArgs):
    """ Train a model from scratch on a data. """
    if config_args.exists():
        outdir_args = config_args.get_outdir_args()
        sys_args = config_args.get_sys_args()
        individual_args = config_args.get_individual_args()
        problem_args = config_args.get_problem_args()
        ga_args = config_args.get_ga_args()

    print_dict_highlighted(vars(ga_args))

    ga_args.crossover = "custom"
    ga_args.mutation = "custom"
    ga_args.selection = "custom"
    ga_args.trainer_name = "custom"
    individual_args.individual_name = "custom"

    population_manager: PopulationManager = PopulationManager(ga_args, individual_args, sys_args=sys_args)
    trainer: GA = GAFactory.from_ga_args(population_manager=population_manager, ga_args=ga_args,
                                         problem_args=problem_args, sys_args=sys_args, outdir_args=outdir_args,
                                         individual_args=individual_args)
    trainer.train()


if __name__ == "__main__":
    train(*parse_args())
