from typing import Callable

from gaggle.arguments.ga_args import GAArgs
from gaggle.operators.selection.selection import Selection
from gaggle.operators.selection.base_selections.truncation_selection import TruncationSelection
from gaggle.operators.selection.base_selections.weighted_selection import WeightedSelection
from gaggle.operators.selection.base_selections.relative_weighted_selection import RelativeWeightedSelection
from gaggle.operators.selection.base_selections.probabilistic_tournament_selection import ProbabilisticTournamentSelection
from gaggle.operators.selection.base_selections.simple_tournament_selection import SimpleTournamentSelection


class SelectionFactory:
    r"""Factory that generates pre-existing available selection operators.
    SelectionFactory.selections stores said selection operators as a dictionary with 
    their name as the key and the uninitialized selection object as the value.

    """
    selections = {
        "truncation": TruncationSelection,
        "weighted": WeightedSelection,
        "relative_weighted": RelativeWeightedSelection,
        "probabilistic_tournament": ProbabilisticTournamentSelection,
        "simple_tournament": SimpleTournamentSelection
    }

    @classmethod
    def get_keys(cls):
        r"""Returns the list of currently registered selections """
        return list(cls.selections.keys())

    @classmethod
    def update(cls, key, selection):
        r"""Add a new selection operator to the dictionary of selections that can be created.

        It is added to SelectionFactory.selections

        Args:
            key: selection name that will be used as the dictionary lookup key
            selection: selection class object, it needs to not be already initialized
        """
        assert isinstance(selection, Callable)
        cls.selections[key] = selection

    @classmethod
    def from_ga_args(cls, ga_args: GAArgs = None) -> Selection:
        r"""Initializes the requested selection from the dictionary of available selections.

        This is done by using the attribute ga_args.selection as
        the lookup key to SelectionFactory.selections.

        Args:
            ga_args: GAArgs object for the current run

        Returns:
            An initialized selection class object.

        """
        ga_args = ga_args if ga_args is not None else GAArgs()
        selection = cls.selections.get(ga_args.selection, None)
        if selection is None:
            raise ValueError(ga_args.selection)
        return selection(ga_args)
