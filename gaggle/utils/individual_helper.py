import torch


def from_gene_pool(gene_pool: dict) -> (torch.Tensor, tuple[list, list]):
    """From a gene_pool generates a flattened tensor that contains all the params of the gene_pool as well
    as metadata to be passed to from_tensor to recreate the gene_pool after any
    modification to the newly created flattened tensor.

    Args:
        gene_pool: dictionary of parameters that defines an Individual.

    Returns:
        Flattened gene tensor and a metadata tuple.
    """
    tensors = []
    gene_sizes = []
    shapes = []
    for key in gene_pool.keys():
        tensor = gene_pool[key]["param"].data.clone().detach()
        shapes.append(tensor.size())
        gene_sizes.append(gene_pool[key]["gene_size"])
        flattened_tensor = tensor.flatten()
        tensors.append(flattened_tensor)
    metadata = (gene_sizes, shapes)

    return torch.cat(tensors, dim=0), metadata


def from_gene_pool_no_metadata(gene_pool: dict) -> torch.Tensor:
    """Same as from_gene_pool but does return the metadata. It is used in cases where we do not care 
    about transforming the tensor back into the gene pool (for example metric computation on the gene pool).

    Args:
        gene_pool:

    Returns:
        Flattened gene tensor.
    """
    tensors = []

    for key in gene_pool.keys():
        tensor = gene_pool[key]["param"].data.clone().detach()
        flattened_tensor = tensor.flatten()
        tensors.append(flattened_tensor)

    return torch.cat(tensors, dim=0)


def from_tensor(gene_pool: dict, tensor: torch.Tensor, metadata: tuple[list, list]) -> dict:
    """Updates the parameters in gene_pool from the flattened tensor tensor inplace.

    Args:
        gene_pool: dictionary of parameters that defines an Individual.
        tensor: the flattened tensor of weights that will be transformed into the gene_pool.
        metadata: the metadata that was acquired when originally running from_gene_pool on gene_pool to get the tensor.

    Returns:
        The modified gene_pool dictionary

    Notes:
        It applies the transformation of the gene_pool inplace on the gene_pool argument (even though it still returns
        it)
    """
    assert len(tensor.size()) == 1  # we want a flattened tensor
    gene_sizes, shapes = metadata
    curr = 0
    for i, key in enumerate(gene_pool.keys()):
        gene_size = gene_sizes[i]
        shape = shapes[i]
        gene_pool[key]["param"].data = torch.unflatten(tensor[curr:curr+gene_size], 0, shape)
        curr += gene_size

    return gene_pool
