import os
import copy
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter


from gaggle.arguments import IndividualArgs, SysArgs
from gaggle.population.individual import Individual
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.utils.special_print import print_highlighted
from gaggle.utils.web import is_valid_url


class NNIndividual(Individual):
    """An Individual whose initial parameters are a torch.nn.Module.

    """
    CONFIG_BASE_MODEL_STATE_DICT = "base_model_state_dict"

    def __init__(self, individual_args: IndividualArgs, sys_args: SysArgs = None, model: nn.Module = None,
                 *args, **kwargs):
        # we first initialize all the variables we need
        super().__init__(individual_args, sys_args)
        if model is None:
            self.model = individual_args.get_base_model(*args, **kwargs).to(self.sys_args.device)
        else:
            self.model = copy.deepcopy(model).to(self.sys_args.device)
        self.hooks = []
        self.return_val = {}
        self.print_flags = {}
        self._debug_mode = False
        self._tick = 0
        if individual_args.model_ckpt is not None:
            self.load()

    def initialize(self):
        if self.individual_args.random_init:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # nn.init.constant_(m.weight, 0.00001)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        return self

    def generate_gene_pool(self):
        gene_pool = {}
        self.genome_size = 0
        for i, m in enumerate(self.model.parameters()):
            gene_size = 1
            for dim in list(m.size()):
                gene_size *= dim
            gene_pool[i] = {"param": m, "gene_size": gene_size}
            self.genome_size += gene_size
        return gene_pool

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.model.parameters()

    def train(self, mode: bool = True):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, x):
        return self.model(x)

    def save(self, outdir_args: OutdirArgs = None) -> dict:
        data = {
            self.CONFIG_INDIVIDUAL_ARGS: self.individual_args,
            self.CONFIG_BASE_MODEL_STATE_DICT: self.model.state_dict()
        }
        if outdir_args is not None:
            folder = outdir_args.get_folder_path()
            fn = os.path.join(folder, f"{self.individual_args.model_name}.pt")
            torch.save(data, fn)
            print_highlighted(f"Saved model at {os.path.abspath(fn)}")
        return data

    def debug(self, mode: bool = True) -> None:
        self._debug_mode = mode

    def debug_tick(self) -> None:
        """ Clears plotting for debugs. """
        self._tick += 1

    def first_time(self, name) -> bool:
        """ Checks if something has been invoked for the first time """
        state = name not in self.print_flags
        self.print_flags[name] = True
        return state

    def load(self, content=None, ckpt=None) -> nn.Module:
        if content is None:
            ckpt = ckpt if ckpt is not None else self.model_args.model_ckpt

            if is_valid_url(ckpt):
                content = torch.hub.load_state_dict_from_url(ckpt, progress=False)
            else:
                content = torch.load(ckpt)

        if IndividualArgs.CONFIG_KEY in content.keys():
            content = content[IndividualArgs.CONFIG_KEY]

        # Hacky part. See if this is a checkpoint to load the base model or to load this model.
        if self.CONFIG_MODEL_ARGS in content.keys():
            self.model_args = content[self.CONFIG_MODEL_ARGS]
            self.model.load_state_dict(content[self.CONFIG_BASE_MODEL_STATE_DICT])
            self.model.eval()
        else:
            # we assume this is just a state dict for the base model
            self.model.load_state_dict(content)
            self.model.eval()
        return self.model