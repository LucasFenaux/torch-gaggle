from gaggle.population import Individual
from gaggle.arguments import SysArgs, IndividualArgs
from gaggle.utils.individual_helper import from_gene_pool_no_metadata
import torch
import torch.nn as nn


class PytorchIndividual(Individual):
    """An Individual whose initial parameters are pytorch tensors.

    """

    def __init__(self, tensors: dict[int: torch.Tensor] = None, individual_args: IndividualArgs = None,
                 sys_args: SysArgs = None):
        super(PytorchIndividual, self).__init__(individual_args, sys_args)
        if tensors is None:
            self.initialize()
        else:
            self.gene_pool = self.generate_gene_pool(tensors)


    def initialize(self, *args, **kwargs) -> nn.Module:
        if self.gene_pool is None:
            low = self.individual_args.param_lower_bound if self.individual_args.param_lower_bound is not None else 0.
            high = self.individual_args.param_upper_bound if self.individual_args.param_upper_bound is not None else 1.

            tensors = {0: torch.empty(self.individual_args.individual_size).uniform_(low, high).clone().detach()}
            self.gene_pool = self.generate_gene_pool(tensors)
        return self

    def generate_gene_pool(self, tensors: dict[int: torch.Tensor], *args,
                           **kwargs) -> dict[int:dict[str: nn.Parameter, str: int]]:
        gene_pool = {}
        self.genome_size = 0
        idx = 0
        for value in tensors.values():
            param = nn.Parameter(value.clone().detach())
            self.register_parameter(str(idx), param)
            gene_size = 1
            for dim in list(param.size()):
                gene_size *= dim
            gene_pool[idx] = {"param": param, "gene_size": gene_size}
            self.genome_size += gene_size
            idx += 1
        self.to(self.sys_args.device)
        return gene_pool

    def forward(self, *args, **kwargs):
        return from_gene_pool_no_metadata(self.get_gene_pool())
