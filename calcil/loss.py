# -*- coding: utf-8 -*-
"""Implementation of Loss class for running gradient-based reconstruction."""

import jax.numpy as jnp
import jax
import functools


def loss_fn_checker(fn):
    """decorator for loss function callable"""
    raise NotImplementedError


class Loss:
    """Loss class that wraps a loss function and its name. It is used to define the loss function of a model. The

    loss function is a callable that takes the following arguments::

        forward_output: output of the forward function of the model
        variables: trainable variables of the model
        input_dict: input dictionary from the data loader
        intermediate: intermediate variables of the model (optional)
        and returns a tuple of (loss, aux_dict). The aux_dict is a dictionary of auxiliary terms that are used for
        logging and debugging. The loss is a scalar value.

    Simple arithmetic operations are defined for the Loss class. For example, if loss_fn1 and loss_fn2 are two Loss
    objects, then loss_fn1 + loss_fn2 is also a Loss object. The loss function of the new Loss object is the sum of the
    loss functions of loss_fn1 and loss_fn2. The weights of the two loss functions are also added together. The same
    applies to multiplication with a scalar.

    Args:
        loss_fn (callable or list of callables): loss function(s)
        name (str or list of str): name(s) of the loss function(s) for logging
        weight (float or list of float): weight(s) of the loss function(s)
        has_intermediates (bool): whether the loss function needs intermediate variables or not
    """
    def __init__(self, loss_fn, name, weight=None, has_intermediates=False):
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

        self.enable_intermediates = has_intermediates

    def get_loss_fn(self):
        def loss_fn(variables, input_dict, forward_fn):
            if self.enable_intermediates:
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
        return Loss(self.loss_fn + other.loss_fn, self.names + other.names, self.weights + other.weights,
                    self.enable_intermediates|other.enable_intermediates)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplementedError('Need scalar value for the multiplication')

        return Loss(self.loss_fn, self.names, [w * scalar for w in self.weights], self.enable_intermediates)

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

