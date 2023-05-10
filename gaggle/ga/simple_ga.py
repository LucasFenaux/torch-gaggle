import os
from typing import Callable, List
import time
import json

import torch
import matplotlib.pyplot as plt

from gaggle.arguments import GAArgs, SysArgs, IndividualArgs, ProblemArgs
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.population.individual import Individual
from gaggle.population.population_manager import PopulationManager
from gaggle.utils.special_print import print_dict_highlighted, print_highlighted
from gaggle.operators import Crossover, Mutation, Selection
from gaggle.ga import GA
from gaggle.problem import Problem


class SimpleGA(GA):
    r"""Implements a Simple Genetic Algorithm following Mitchell.

    """
    def __init__(self, population_manager: PopulationManager = None, ga_args: GAArgs = None,
                 selection: Selection = None, crossover: Crossover = None, mutation: Mutation = None,
                 problem_args: ProblemArgs = None, sys_args: SysArgs = None, outdir_args: OutdirArgs = None,
                 individual_args: IndividualArgs = None, problem: Problem = None):
        super(SimpleGA, self).__init__(population_manager, ga_args, selection, crossover, mutation, problem_args,
                                       sys_args, outdir_args, individual_args, problem)
        self.best = 0
        self.saved_metrics = {"train_metrics": {"best": [],
                                                "avg": [],
                                                "worst": [],
                                                "generation": [],
                                                "time_taken": []
                                                },
                              "test_metrics": {"best": [],
                                               "avg": [],
                                               "worst": [],
                                               "generation": [],
                                               "time_taken": []
                                               }
                              }

        self.window_size = 10
        self.metrics_to_plot = ["best", "avg", "worst"]

    def save_individual(self, individual: Individual, fitness):
        """Saves the individual provided as well as its fitness to the folder path specified in the outdir_args as
        'best.pt'.

        Args:
            individual:
            fitness:

        Returns:

        """
        if (self.ga_args.save_every_epoch or (fitness > self.best)):
            data = {
                "fitness": fitness,
                GAArgs.CONFIG_KEY: vars(self.ga_args),
                IndividualArgs.CONFIG_KEY: individual.save(),
            }
            fn = os.path.join(self.outdir_args.create_folder_name(), f"best.pt")
            torch.save(data, fn)
            print_highlighted(f"New best: {fitness:.2f}%>{self.best:.2f}%. Saved at '{os.path.abspath(fn)}'")
            self.best = fitness

    @torch.no_grad()
    def get_fitness_metric(self, fitness: dict[int:float], save: bool = False, mode: str = "train"):
        """
        Outputs basic fitness metrics for a population, like avg fitness, best & worst fitness
        :param fitness: dictionary of fitness
        :param save: whether to save the metrics
        :param mode: what to save
        :return: metrics: dictionary of metrics
        """
        metrics = {}
        # best
        best_idx = max(fitness, key=fitness.get)
        best_fitness = fitness[best_idx]
        metrics["best"] = [best_idx, best_fitness]
        # worst
        worst_idx = min(fitness, key=fitness.get)
        worst_fitness = fitness[worst_idx]
        metrics["worst"] = [worst_idx, worst_fitness]
        # average
        fitness_list = list(fitness.values())
        avg_fitness = sum(fitness_list)/len(fitness_list)
        metrics["avg"] = [avg_fitness]
        if save:
            if mode == "train":
                self.saved_metrics["train_metrics"]["best"].append(best_fitness)
                self.saved_metrics["train_metrics"]["avg"].append(avg_fitness)
                self.saved_metrics["train_metrics"]["worst"].append(worst_fitness)
            elif mode == "eval":
                self.saved_metrics["test_metrics"]["best"].append(best_fitness)
                self.saved_metrics["test_metrics"]["avg"].append(avg_fitness)
                self.saved_metrics["test_metrics"]["worst"].append(worst_fitness)
            else:
                raise NotImplementedError

        return metrics

    def display_metrics(self, display_train: bool = True, display_test: bool = True):
        """Displays the metrics computed and stored in self.saved_metrics. Only displays and saves
        the metrics whose keys are in self.metrics_to_plot. The graphs are also saved to a file in the output folder
        specified in outdir_args.

        Args:
            display_train:
            display_test:

        Returns:

        """
        if display_train:
            plt.figure()
            plt.clf()
            plt.title('Training Metrics')
            plt.xlabel('Generations')
            plt.ylabel('Fitness')
            moving_averages = {}
            for key in self.saved_metrics["train_metrics"].keys():
                if key in self.metrics_to_plot:
                    moving_averages[key] = []
                    for i in range(len(self.saved_metrics["train_metrics"][key]) - self.window_size):
                        moving_averages[key].append(sum(self.saved_metrics["train_metrics"][key][i: i+self.window_size])/
                                                    self.window_size)
                    plt.plot(list(range(len(moving_averages[key]))), moving_averages[key], label=key)
            fn = os.path.join(self.outdir_args.create_folder_name(), f"training_metrics.png")
            plt.legend()
            plt.savefig(fn)
            plt.show()

        if display_test:
            plt.figure()
            plt.clf()
            plt.title('Eval Metrics')
            plt.xlabel('Generations')
            plt.ylabel('Fitness')
            moving_averages = {}
            for key in self.saved_metrics["test_metrics"].keys():
                if key in self.metrics_to_plot:
                    moving_averages[key] = []
                    for i in range(len(self.saved_metrics["test_metrics"][key]) - self.window_size):
                        moving_averages[key].append(sum(self.saved_metrics["test_metrics"][key][i: i+self.window_size])/
                                                    self.window_size)
                    plt.plot(list(range(len(moving_averages[key]))), moving_averages[key], label=key)
            fn = os.path.join(self.outdir_args.create_folder_name(), f"testing_metrics.png")
            plt.legend()
            plt.savefig(fn)
            plt.show()

    def save_metrics(self, save_train: bool = True, save_test: bool = True):
        """Saves the metrics computed and stored in self.saved_metrics. The metrics are saved to files in the output
        folder specified in outdir_args.

        Args:
            save_train:
            save_test:

        Returns:

        """
        if save_train:
            train_fn = os.path.join(self.outdir_args.create_folder_name(), f"training_metrics.json")
            with open(train_fn, "w") as f:
                json.dump(self.saved_metrics["train_metrics"], f)

        if save_test:
            eval_fn = os.path.join(self.outdir_args.create_folder_name(), f"test_metrics.json")
            with open(eval_fn, "w") as f:
                json.dump(self.saved_metrics["test_metrics"], f)

    def test(self, save: bool = True):
        """Tests population by computing the fitness and its associated metrics without updating the population manager.

        Args:
            save:

        Returns:

        """
        start_time = time.time()
        self.population_manager.eval()

        test_fitness = self.problem.evaluate_population(self.population_manager, use_freshness=False, update_manager=False,
                                                        train=False)
        test_metrics = self.get_fitness_metric(test_fitness, save=save, mode="eval")
        if self.outdir_args is not None:
            self.save_individual(self.population_manager.get_individual(test_metrics["best"][0]), test_metrics["best"][1])

        test_metrics["time_taken"] = time.time() - start_time
        self.saved_metrics["test_metrics"]["time_taken"].append(test_metrics["time_taken"])

        print("Test Metrics")
        print_dict_highlighted(test_metrics)

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
        self.population_manager = self.mutation_fn.mutate_pop(self.population_manager)
        return train_fitness

    def train(self, test: bool = True,
              callbacks: List[Callable] = None,
              display_train_metrics: bool = True,
              display_test_metrics: bool = True):
        """Call to begin the training process of the population using the arguments stored in this SimpleGA object.

        Args:
            test:
            callbacks:
            display_train_metrics:
            display_test_metrics:

        Returns:

        """
        print(f"Genome size: {self.population_manager.get_gene_count():.3e} params")

        print_dict_highlighted(vars(self.ga_args))

        if callbacks is None:
            callbacks = []

        for generation in range(self.ga_args.generations):
            start_time = time.time()
            train_fitness = self.train_one_generation()
            time_taken = time.time()
            metrics = self.get_fitness_metric(train_fitness, save=True, mode="train")
            metrics["generation"] = f"{generation+1}/{self.ga_args.generations}"
            metrics["time_taken"] = time_taken - start_time
            print_dict_highlighted(metrics)
            # we add the generation and time taken to the saved metrics
            self.saved_metrics["train_metrics"]["generation"].append(generation+1)
            self.saved_metrics["train_metrics"]["time_taken"].append(metrics["time_taken"])
            if test and self.ga_args.eval_every_generation \
                    and generation % self.ga_args.eval_every_generation == 0:
                self.test()
                self.saved_metrics["test_metrics"]["generation"].append(generation+1)
            for callback in callbacks:
                callback(generation)

        if test:
            self.test()

        if display_test_metrics or display_train_metrics:
            self.display_metrics(display_train=display_train_metrics, display_test=display_test_metrics)
