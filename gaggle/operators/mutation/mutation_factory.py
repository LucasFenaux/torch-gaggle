from typing import Callable

from gaggle.arguments.ga_args import GAArgs
from gaggle.operators.mutation.mutation import Mutation
from gaggle.operators.mutation.base_mutations.normal_mutation import NormalMutation
from gaggle.operators.mutation.base_mutations.uniform_mutation import UniformMutation


class MutationFactory:
    r"""Factory that generates pre-existing available mutation operators.
    MutationFactory.mutations stores said mutation operators as a dictionary with 
    their name as the key and the uninitialized mutations object as the value.

    """
    mutations = {
        "normal": NormalMutation,
        "uniform": UniformMutation
    }

    @classmethod
    def get_keys(cls):
        r"""Returns the list of currently registered mutations """
        return list(cls.mutations.keys())

    @classmethod
    def update(cls, key, mutation):
        r"""Add a new mutation operator to the dictionary of mutations that can be created.

        It is added to MutationFactory.mutations

        Args:
            key: mutation name that will be used as the dictionary lookup key
            mutation: mutation class object, it needs to not be already initialized
        """
        assert isinstance(mutation, Callable)
        cls.mutations[key] = mutation

    @classmethod
    def from_ga_args(cls, ga_args: GAArgs = None) -> Mutation:
        r"""Initializes the requested mutation from the dictionary of available mutations.

        This is done by using the attribute ga_args.mutation as
        the lookup key to MutationFactory.mutations.

        Args:
            ga_args: GAArgs object for the current run

        Returns:
            An initialized mutation class object.

        """
        ga_args = ga_args if ga_args is not None else GAArgs()
        mutation = cls.mutations.get(ga_args.mutation, None)
        if mutation is None:
            raise ValueError(ga_args.mutation)
        return mutation(ga_args)
