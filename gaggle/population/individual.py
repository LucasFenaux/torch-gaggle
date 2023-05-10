from abc import abstractmethod

import torch
import torch.nn as nn
import os
from gaggle.arguments import SysArgs, IndividualArgs
from gaggle.utils.individual_helper import from_gene_pool
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.utils.special_print import print_highlighted


class Individual(nn.Module):
    """An object that stores a single candidate solution to the given problem (aka a chromosome).
    This is the parent class for any other individual type.
    The gene_pool field stores the entire chromosome formatted as pytorch model parameters.
    """
    CONFIG_INDIVIDUAL_ARGS = "individual_args_config"
    CONFIG_GENE_POOL = "gene_pool_config"

    def __init__(self, individual_args: IndividualArgs = None, sys_args: SysArgs = None):
        super().__init__()
        # gene_pool should be a dict[int: dict["param": nn.Parameter, "gene_size": int]]
        self.gene_pool = None
        self.genome_size = None
        self.sys_args = sys_args if sys_args is not None else SysArgs()
        self.individual_args = individual_args if individual_args is not None else IndividualArgs()

    @abstractmethod
    def generate_gene_pool(self, *args, **kwargs) -> dict[int:dict[str: nn.Parameter, str: int]]:
        """ Should return a dictionary of dictionaries where the inner dictionary has a "param" and a "size" key
        for each of the nn.Parameters in the dictionary """
        raise NotImplementedError

    def get_gene_pool(self):
        """Returns the gene pool of the individual if it already exists, otherwise generates it then returns it.

        Returns:
            The individual's gene pool as a dictionary

        """
        if self.gene_pool is None:
            self.gene_pool = self.generate_gene_pool()
        if self.genome_size is None:
            self.get_genome_size()
        return self.gene_pool

    def get_genome_size(self):
        """Returns the genome size of the individual if it already exists, otherwise generates it then returns it.
        If the gene pool has not been generated when this is called, it will generate it.

        Returns:
            The individual's genome size as an int.

        """
        if self.gene_pool is None:
            self.get_gene_pool()
        if self.genome_size is None:
            self.genome_size = 0
            for key, value in self.gene_pool.items():
                self.genome_size += self.gene_pool[key]["gene_size"]

        return self.genome_size

    def initialize(self, *args, **kwargs) -> nn.Module:
        """Represents the initialization rule for an individual. It should modify the gene pool.

        Args:
            *args:
            **kwargs:

        Returns:
            self
        """
        return self

    @staticmethod
    def clip(tensor, lower_bound = None, upper_bound = None):
        """torch.clip wrapper in case the individual created does not use a type support by torch.clip. If that is the
        case, the individual can overwrite this clip method to the desired behavior.

        Args:
            tensor: torch tensor to clip
            lower_bound: min value
            upper_bound: max value

        Returns:

        """
        return torch.clip(tensor, min=lower_bound, max=upper_bound)

    def apply_bounds(self, lower_bound=None, upper_bound=None):
        """Apply parameter bounds.

        Args:
            lower_bound:
            upper_bound:

        Returns:

        """
        self.get_gene_pool()
        for key, value in self.gene_pool.items():
            param = value["param"]
            param.data = self.clip(param.data, lower_bound=lower_bound, upper_bound=upper_bound)

    def save(self, outdir_args: OutdirArgs = None) -> dict:
        """Saves the individual to a file in the path provided by outdir_args.

        Args:
            outdir_args:

        Returns:

        """
        data = {
            self.CONFIG_INDIVIDUAL_ARGS: self.individual_args,
            self.CONFIG_GENE_POOL: self.gene_pool
        }
        if outdir_args is not None:
            folder = outdir_args.get_folder_path()
            fn = os.path.join(folder, f"{self.individual_args.individual_name}.pt")
            torch.save(data, fn)
            print_highlighted(f"Saved model at {os.path.abspath(fn)}")
        return data

    def forward(self, *args, **kwargs):
        """By default we returned a flattened tensor of the model parameters.

        Notes:
            We do not return the metadata that can be used to reconstruct the gene_pool dictionary as this should
            not be used to modify the parameters directly (the tensor does not link back to the parameters).

        Args:
            *args:
            **kwargs:

        Returns:
            A flattened pytorch tensor of length self.genome_size.
        """
        return from_gene_pool(self.get_gene_pool())[0]

    def __len__(self):
        return self.get_genome_size()

