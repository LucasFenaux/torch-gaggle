from gaggle.arguments.sys_args import SysArgs
from gaggle.arguments.individual_args import IndividualArgs
from gaggle.population.base_individuals.nn_individual import NNIndividual
from gaggle.population.base_individuals.numpy_individual import NumpyIndividual
from gaggle.population.base_individuals.rl_individual import RLIndividual
from gaggle.population.base_individuals.pytorch_individual import PytorchIndividual


class IndividualFactory:
    r"""Factory that generates pre-existing available Individuals.
    IndividualFactory.individuals stores said Individuals as a dictionary with their name as key 
    and the uninitialized Individual object as value.
    """

    individuals = {
        "nn": NNIndividual,
        "numpy": NumpyIndividual,
        "rl": RLIndividual,
        "pytorch": PytorchIndividual
    }

    @classmethod
    def update(cls, key, individual):
        r"""Add a new Individual to the dictionary of Individuals that can be created.

        It is added to IndidividualFactory.individuals

        Args:
            key: dataset name that will be used as the dictionary lookup key
            individual: Individual class object, it needs to not be already initialized

        """
        cls.individuals[key] = individual

    @classmethod
    def from_individual_args(cls, individual_args: IndividualArgs = None, sys_args: SysArgs = None, *args, **kwargs):
        r"""Initializes the requested Individual from the dictionary of available Individuals.

        This is done by using the attribute individual_args.individual_name as
        the lookup key to IndividualFactory.individuals.
        Args:
            individual_args:
            sys_args:
            *args:
            **kwargs:

        Returns:
            Initialized Individual object.
        """
        individual_args = individual_args if individual_args is not None else IndividualArgs()
        individual = cls.individuals.get(individual_args.individual_name, None)
        if individual is None:
            raise ValueError(individual_args.individual_name)
        return individual(individual_args=individual_args, sys_args=sys_args, *args, **kwargs)
