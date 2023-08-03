"""
HybridGA code for this project that is to be added to the main gaggle package once enough testing has been performed.
"""

from typing import Callable, List

import torch
import torch.nn as nn

from gaggle.arguments import GAArgs, SysArgs, IndividualArgs, ProblemArgs
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.population import NNIndividual
from gaggle.population.population_manager import PopulationManager
from gaggle.operators import Crossover, Mutation, Selection
from gaggle.ga import SimpleGA
from gaggle.problem import Problem
from gaggle.utils import MetricLogger, SmoothedValue
import time
from tqdm import tqdm
import os


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def train_one_epoch(model, criterion, optimizer, data_loader, device, model_num, ga_args: GAArgs):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Model #: [{model_num}]"
    pbar = tqdm(data_loader)

    for i, (image, target) in enumerate(pbar):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        if ga_args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), ga_args.clip_grad_norm)
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        pbar.set_description(header + metric_logger.__str__())


class HybridGA(SimpleGA):
    r"""Implements a Hybrid Genetic Algorithm that by default uses Gradient Descent as its additional operator.
    """
    def __init__(self, population_manager: PopulationManager = None, ga_args: GAArgs = None,
                 selection: Selection = None, crossover: Crossover = None, mutation: Mutation = None,
                 problem_args: ProblemArgs = None, sys_args: SysArgs = None, outdir_args: OutdirArgs = None,
                 individual_args: IndividualArgs = None, problem: Problem = None, criterion: Callable = None,
                 optimizer_fn: Callable = None, lr_scheduler_fn: Callable = None,
                 mutation_scheduler_fn: Callable = None):
        super(HybridGA, self).__init__(population_manager, ga_args, selection, crossover, mutation, problem_args,
                                       sys_args, outdir_args, individual_args, problem)
        self.data_loader = self.setup_dataloader()
        self.criterion = criterion if criterion is not None else self.setup_criterion()
        self.optimizer_fn = optimizer_fn if optimizer_fn is not None else self.setup_optimizer_fn()
        self.window_size = 2
        # we will only change the arg in the mutation function so it should have little to no trickle down effect but
        self.original_mutation_std = self.ga_args.mutation_std  # we store it here since we'll have to modify the argument value due to how mutation operates

        # we setup a dummy optimizer for the scheduler to attach to so we can update the learning rate for all future
        # optimizers we create by referencing to this optimizer's learning rate
        self.dummy_lr_optimizer = self.optimizer_fn({torch.nn.Parameter(torch.Tensor([0.]))}, lr=self.ga_args.lr)
        self.dummy_mutation_optimizer = self.optimizer_fn({torch.nn.Parameter(torch.Tensor([0.]))},
                                                          lr=self.ga_args.mutation_std)

        self.lr_scheduler_fn = lr_scheduler_fn if lr_scheduler_fn is not None else self.setup_scheduler_fn(
            self.ga_args.lr_scheduler)
        if self.lr_scheduler_fn is not None:
            self.lr_scheduler = self.lr_scheduler_fn(self.dummy_lr_optimizer, self.ga_args.generations)

        self.mutation_scheduler_fn = mutation_scheduler_fn if mutation_scheduler_fn is not None else \
            self.setup_scheduler_fn(self.ga_args.mutation_std_scheduler)
        if self.mutation_scheduler_fn is not None:
            self.mutation_std_scheduler = self.mutation_scheduler_fn(self.dummy_mutation_optimizer,
                                                                     self.ga_args.generations)

    def setup_dataloader(self):
        dataset_train = self.problem.train_dataset
        data_loader = torch.utils.data.DataLoader(dataset_train, num_workers=self.sys_args.num_workers,
                                                  shuffle=True, batch_size=self.ga_args.hybrid_batch_size)
        return data_loader

    def setup_optimizer_fn(self):
        if self.ga_args.optimizer == "SGD":
            optimizer_fn = torch.optim.SGD
        elif self.ga_args.optimizer == "Adam":
            optimizer_fn = torch.optim.Adam
        else:
            raise NotImplementedError

        return optimizer_fn

    def setup_criterion(self):
        if self.ga_args.criterion == "CE":
            return torch.nn.CrossEntropyLoss()
        elif self.ga_args.criterion == "BCE":
            return torch.nn.BCELoss()
        else:
            raise NotImplementedError

    @staticmethod
    def setup_scheduler_fn(sched_name):
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        else:
            return None

    def get_optimizer(self, individual: NNIndividual):
        if self.ga_args.optimizer == "SGD" or self.ga_args.optimizer == "Adam":
            return self.optimizer_fn(individual.parameters(), lr=self.dummy_lr_optimizer.param_groups[0]['lr'],
                                     weight_decay=self.ga_args.weight_decay, momentum=self.ga_args.momentum)
        else:
            raise NotImplementedError

    def optimize_individual(self, individual: NNIndividual, individual_num: int):
        # TODO: add line that guarantees data_loader epoch reset to make sure any bug/problem in the training code does
        # TODO: not leave a data_loader epoch midway through
        optimizer = self.get_optimizer(individual)
        train_one_epoch(model=individual, criterion=self.criterion, optimizer=optimizer, data_loader=self.data_loader,
                        device=self.sys_args.device, ga_args=self.ga_args, model_num=individual_num)
        return individual

    def optimize_pop(self, population_manager: PopulationManager):
        # we get the list of population indices
        can_be_optimized = list(self.population_manager.get_population().keys())
        if not self.ga_args.opt_protected:
            protected_list = population_manager.get_protected()
            for protected in protected_list:
                can_be_optimized.remove(protected)

        # to_optimize = []
        # for ind_id in can_be_optimized:
        #     p = random.random()
        #     if p < self.ga_args.opt_chance:
        #         # selected for optimization
        #         to_optimize.append(ind_id)

        mask = torch.rand(len(can_be_optimized), dtype=torch.float,
                          device=torch.device("cpu")) < self.ga_args.opt_chance
        indices = torch.nonzero(mask, as_tuple=True)[0]

        population = population_manager.get_population()
        new_freshness = population_manager.get_freshness()

        # we optimize!
        for i, to_opt in enumerate(indices.tolist()):
            population[to_opt] = self.optimize_individual(population[to_opt], i)
            new_freshness[to_opt] = True

        population_manager.update_population(population, new_freshness)
        # in case we need to enforce parameter value bounds on the freshly mutated samples. Since it only applies to
        # mutated samples we don't have to update the freshness
        population_manager.apply_bounds(indices)

        return population_manager

    def train_one_generation(self):
        """
        Standard one generation GA pipeline
        """
        self.population_manager.train()
        train_fitness = self.problem.evaluate_population(self.population_manager,
                                                         use_freshness=self.ga_args.use_freshness, update_manager=True,
                                                         train=True)
        self.population_manager = self.selection_fn.select_all(self.population_manager,
                                                               self.crossover_fn.mates_per_crossover,
                                                               self.crossover_fn.children_per_crossover)
        self.population_manager = self.crossover_fn.crossover_pop(self.population_manager)
        if self.ga_args.opt_before_mutation:
            self.population_manager = self.optimize_pop(self.population_manager)
            self.population_manager = self.mutation_fn.mutate_pop(self.population_manager)
        else:
            self.population_manager = self.mutation_fn.mutate_pop(self.population_manager)
            self.population_manager = self.optimize_pop(self.population_manager)

        # we take a step to update the learning rate
        if self.lr_scheduler:
            self.lr_scheduler.step()

        if self.mutation_std_scheduler:
            self.mutation_std_scheduler.step()
            # we now propagate that change
            self.update_mutation_std()

        return train_fitness

    def update_mutation_std(self):
        assert self.mutation_std_scheduler is not None
        self.mutation_fn.ga_args.mutation_std = self.dummy_mutation_optimizer.param_groups[0]['lr']

    def save_population_snapshot(self):
        print("saving population")
        fn = os.path.join(self.outdir_args.create_folder_name(), "population_and_fitness.pt")
        data = {"population": self.population_manager.get_population(),
                "fitness": self.population_manager.get_fitness()}
        torch.save(data, fn)

    def train(self,
              test: bool = True,
              callbacks: List[Callable] = None,
              display_train_metrics: bool = True,
              display_test_metrics: bool = True):
        super(HybridGA, self).train(test, callbacks, display_train_metrics, display_test_metrics)
        self.save_population_snapshot()
