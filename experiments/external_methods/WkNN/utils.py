import torch

import numpy as np
from scipy.stats import norm

from DSRegressor.DSRule import DSRuleCategorical, DSRuleLess, DSRuleBetween, DSRuleMore, DSRulePositive, DSRuleNegative


def get_prod(values):
    square_matrix = values.unsqueeze(2).repeat(1, 1, values.size(1))

    for ind in range(square_matrix.size(0)):
        torch.diagonal(square_matrix[ind, :, :]).fill_(0)
    return (1 - square_matrix).prod(dim=1)


def is_categorical(arr, max_cat=3):
    return len(np.unique(arr[~np.isnan(arr)])) <= max_cat


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


def generate_statistic_single_rules(X, breaks=2, column_names=None, mass_vector=None):
    """
    Populates the model with attribute-independant rules based on statistical breaks.
    In total this method generates No. attributes * (breaks + 1) rules
    :param X: Set of inputs (can be the same as training or a sample)
    :param breaks: Number of breaks per attribute
    :param column_names: Column attribute names
    """
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    brks = norm.ppf(np.linspace(0, 1, breaks + 2))[1:-1]
    rules = []
    if column_names is None:
        column_names = ["X[%d]" % i for i in range(len(mean))]
    for i in range(len(mean)):
        if is_categorical(X[:, i]):
            categories = np.unique(X[:, i][~np.isnan(X[:, i])])
            for cat in categories:
                rules.append(
                    DSRuleCategorical(lambda x, i=i, k=cat: x[i] == k, "%s = %s" % (column_names[i], str(cat)),
                                      var=column_names[i], value=cat, mass=mass_vector[i]))
        else:
            # First rule
            v = mean[i] + std[i] * brks[0]
            rules.append(DSRuleLess(lambda x, v=v, i=i: x[i] < v, "%s < %.4f" % (column_names[i], v), inf=v,
                                    var=column_names[i], mass=mass_vector[i]))
            # Mid rules
            for j in range(1, len(brks)):
                vl = v
                v = mean[i] + std[i] * brks[j]
                rules.append(DSRuleBetween(lambda x, vl=vl, v=v, i=i: vl < x[i] < v,
                                           "%.4f < %s < %.4f" % (vl, column_names[i], v), inf=vl, sup=v,
                                           var=column_names[i], mass=mass_vector[i]))
            # Last rule
            rules.append(DSRuleMore(lambda x, v=v, i=i: x[i] > v, "%s > %.3f" % (column_names[i], v), sup=v,
                                    var=column_names[i], mass=mass_vector[i]))
    return rules


def generate_mult_pair_rules(X, column_names=None, include_square=False, mass_vector=None):
    """
    Populates the model with with rules combining 2 attributes by their multipication, adding both positive
    and negative rule. In total this method generates (No. attributes)^2 rules
    :param X: Set of inputs (can be the same as training or a sample)
    :param column_names: Column attribute names
    :param include_square: Includes rules comparing the same attribute (ie x[i] * x[i])
    """
    mean = np.nanmedian(X, axis=0)
    rules = []
    if column_names is None:
        column_names = ["X[%d]" % i for i in range(len(mean))]

    offset = 0 if include_square else 1

    for i in range(len(mean)):
        for j in range(i + offset, len(mean)):
            # mk = mean[i] * mean[j]
            rules.append(DSRulePositive(lambda x, v1=mean[i], v2=mean[j], i=i, j=j: (x[i] - v1) * (x[j] - v2) > 0,
                                        "Positive %s - %.3f, %s - %.3f" % (
                                            column_names[i], mean[i], column_names[j], mean[j]), mass=mass_vector[i]))
            rules.append(DSRuleNegative(lambda x, v1=mean[i], v2=mean[j], i=i, j=j: (x[i] - v1) * (x[j] - v2) < 0,
                                        "Negative %s - %.3f, %s - %.3f" % (
                                            column_names[i], mean[i], column_names[j], mean[j]), mass=mass_vector[i]))
    return rules


def select_top_masses(mass_tensor, top=0.2):
    mass_tensor, sorted_indices = mass_tensor.sort(descending=True)
    mass_tensor = mass_tensor[:, :int(top * mass_tensor.size(1))]
    sorted_indices = sorted_indices[:, :int(top * sorted_indices.size(1))]
    return mass_tensor[0], sorted_indices[0]

