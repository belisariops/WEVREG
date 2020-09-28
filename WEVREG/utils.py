import torch


def get_prod(values):
    square_matrix = values.unsqueeze(2).repeat(1, 1, values.size(1))

    for ind in range(square_matrix.size(0)):
        torch.diagonal(square_matrix[ind, :, :]).fill_(0)
    return (1 - square_matrix).prod(dim=1)


def get_product_not_i(tensor):
    tensor_size = tensor.size(1)
    tensor_vector_size = tensor.size(0)
    val = tensor.repeat(1, tensor_size, 1)
    val = val.view(tensor_size, tensor_vector_size, tensor_size)
    mask = torch.ones(1)
    for index, tensor in enumerate(val):
        ones = torch.ones(len(tensor))
        phi = torch.zeros_like(tensor)
        phi[:, index] = ones
        if index == 0:
            mask = phi
        else:
            mask = torch.cat((mask, phi))
    mask = mask.view_as(val)
    return mask + (1. - mask) * val
