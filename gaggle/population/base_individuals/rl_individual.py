import torch
import torch.nn as nn

from gaggle.arguments import IndividualArgs, SysArgs
from gaggle.population.base_individuals.nn_individual import NNIndividual


class RLIndividual(NNIndividual):
    """NNIndividual wrapper that adds an argmax to the prediction as RL problem usually require the output of the model
    to be an action rather than logits.

    """
    def __init__(self, individual_args: IndividualArgs, sys_args: SysArgs = None, model: nn.Module = None, *args,
                 **kwargs):
        super(RLIndividual, self).__init__(individual_args, sys_args, model, *args, **kwargs)

    def forward(self, x):
        return torch.argmax(super(RLIndividual, self).forward(x))
