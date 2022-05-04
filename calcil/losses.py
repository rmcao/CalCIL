# Description: Implementation of common loss functions
#
# Written by Ruiming Cao on October 08, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from flax import linen as nn

import types
import functools


def loss_fn_checker(fn):
    """decorator for loss function callable"""
    raise NotImplementedError


class Loss:

    def __init__(self, loss_fn, name, weight=None):
        if isinstance(loss_fn, list):
            self.loss_fn = loss_fn
        else:
            self.loss_fn = [loss_fn]

        if weight is None:
            self.weights = [1.0]
        elif isinstance(weight, list):
            self.weights = weight
        else:
            self.weights = [weight]

        if isinstance(name, list):
            self.names = name
        else:
            self.names = [name]

    def get_loss_fn(self, enable_intermediate=False):
        def loss_fn(variables, input_dict, forward_fn):
            if enable_intermediate:
                forward_output, states = forward_fn(variables, input_dict, mutable = ['intermediates'])
                intermediates = states['intermediates']
            else:
                forward_output = forward_fn(variables, input_dict)
                intermediates = None

            l = 0.0
            aux = {}
            for i in range(len(self.weights)):
                cur_l = self.loss_fn[i](forward_output, variables, input_dict, intermediates) * self.weights[i]
                aux[self.names[i]] = cur_l
                l += cur_l
            aux['total_loss'] = l
            return l, aux
        return loss_fn

    def __add__(self, other):
        if not isinstance(other, Loss):
            raise NotImplementedError('Operands in loss addition have to be objects from Loss class.')
        return Loss(self.loss_fn + other.loss_fn, self.names + other.names, self.weights + other.weights)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplementedError('Need scalar value for the multiplication')

        return Loss(self.loss_fn, self.names, [w * scalar for w in self.weights])

    __rmul__ = __mul__


def get_l2_loss(input_key: str):

    def loss_l2_fn(forward_output, variables, input_dict, intermediate):
        loss_l2 = ((input_dict[input_key] - forward_output) ** 2).mean()
        return loss_l2

    return loss_l2_fn


def get_weight_l2_reg():
    def loss_l2_weight(forward_output, variables, input_dict, intermediate):
        def tree_sum_fn(fn):
            return jax.tree_util.tree_reduce(
                lambda x, y: x + fn(y), variables, initializer=0)
        loss_l2_weight = tree_sum_fn(lambda z: jnp.sum(z ** 2)) / tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape)))
        return loss_l2_weight
    return loss_l2_weight

