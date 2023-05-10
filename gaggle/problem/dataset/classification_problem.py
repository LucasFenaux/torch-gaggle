from gaggle.problem.problem import Problem
from gaggle.population import Individual, PopulationManager
from gaggle.arguments import ProblemArgs, SysArgs
from gaggle.problem.dataset import DatasetFactory
from gaggle.utils.smooth_value import SmoothedValue
from gaggle.utils.metrics import accuracy
import torch


class ClassificationProblem(Problem):
    """A Problem that represents a standard Machine Learning classification problem. It stores the associated
    training and validation dataset. Population evaluation optimized for GPU by default to speed up training.
    To create a classification problem with a custom dataset, register said dataset in the DatasetFactory.

    """
    def __init__(self, problem_args: ProblemArgs = None, sys_args: SysArgs = None):
        super(ClassificationProblem, self).__init__(problem_args, sys_args)
        self.train_dataset = DatasetFactory.from_problem_args(problem_args, train=True, sys_args=sys_args)
        self.train_data, self.train_transforms = self.train_dataset.get_data_and_transform()
        if self.problem_args.batch_size == -1:
            # this means use the entire dataset without batching
            self.problem_args.batch_size = self.train_data[0].size(0)

        if self.problem_args.batch_size == self.train_data[0].size(0):
            # we move everything to the gpu and let it live on the gpu
            print(f"Batching is not necessary, will store the entire data on device: {sys_args.device}")
            self.train_data = (self.train_data[0].to(self.sys_args.device), self.train_data[1].to(
                self.sys_args.device))

        self.eval_dataset = DatasetFactory.from_problem_args(problem_args, train=False, sys_args=sys_args)
        self.eval_data, self.eval_transforms = self.eval_dataset.get_data_and_transform()
        if self.problem_args.eval_batch_size == -1:
            self.problem_args.eval_batch_size = self.eval_data[0].size(0)

        self.current_batch = None
        self.fitness_function = accuracy

    @torch.no_grad()
    def evaluate_population(self, population_manager: PopulationManager,
                            use_freshness: bool = True, update_manager: bool = True, train: bool = True,
                            *args, **kwargs) -> dict[int: float]:
        """Population evaluation optimized for GPU by default to speed up training. Should only be modified if
        specific custom behavior is desired. It is usually not recommend to modify this function.

        Args:
            population_manager:
            use_freshness:
            update_manager:
            train:
            *args:
            **kwargs:

        Returns:
            The dictionary of individual fitnesses
        """
        all_data = self.train_data if train else self.eval_data
        transforms = self.train_transforms if train else self.eval_transforms
        batch_size = self.problem_args.batch_size if train else self.problem_args.eval_batch_size
        num_inputs = all_data[0].size(0)

        fitness = {}
        for i in range(population_manager.population_size):
            if population_manager.is_fresh(i) and use_freshness:
                fitness[i] = SmoothedValue()
            elif not use_freshness:
                fitness[i] = SmoothedValue()

        num_batches = num_inputs // batch_size
        rest = num_inputs % batch_size
        for j in range(num_batches):
            data = all_data[0][j * batch_size:(j + 1) * batch_size].to(self.sys_args.device)
            data = transforms(data)
            targets = all_data[1][j * batch_size:(j + 1) * batch_size].to(self.sys_args.device)
            self.current_batch = (data, targets)
            for k in list(fitness.keys()):
                fitness[k].update(self.evaluate(population_manager.get_individual(k), *args, **kwargs), n=batch_size)

        if rest > 0:
            data = transforms(all_data[0][-rest:].to(self.sys_args.device))
            targets = all_data[1][-rest:].to(self.sys_args.device)
            self.current_batch = (data, targets)
            for l in list(fitness.keys()):
                fitness[l].update(self.evaluate(population_manager.get_individual(l), *args, **kwargs), n=batch_size)

        for m in list(fitness.keys()):
            fitness[m] = fitness[m].global_avg
            if update_manager:
                population_manager.set_individual_fitness(m, fitness[m])
                if use_freshness:
                    population_manager.set_freshness(m, False)

        if train and use_freshness:
            return population_manager.get_fitness()

        return fitness

    @torch.no_grad()
    def evaluate(self, individual: Individual, train: bool = True, *args, **kwargs) -> float:
        """Evaluates an individual on the current batch of data.

        Args:
            individual:
            train: whether we are currently training or performing an inference.
            *args:
            **kwargs:

        Returns:

        """
        if train:
            individual.train()
        else:
            individual.eval()
        x, y = self.current_batch
        x, y = x.to(self.sys_args.device), y.to(self.sys_args.device)
        y_pred = individual(x)
        return self.fitness_function(y_pred, y).cpu().item()
