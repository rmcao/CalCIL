# Description:
#  
# Written by Ruiming Cao on January 04, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import functools

import jax.numpy as jnp
import jax
from flax import linen as nn
import flax.core


def _rgetattr(obj, attr, *args):
    """Recursive getattr. from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def _var_retrieve(variables, vid, new_value=None):
    """Retrieve value from vid or replace the value if a new value is supplied."""
    list_keys = vid.split('.')
    last_key = list_keys[-1]

    var = variables
    for k in list_keys[:-1]:
        var = var[k]

    if new_value is not None:
        var[last_key] = new_value
        return variables
    else:
        return var[last_key]


def var_list(variables):
    """Output a list of unique identifiers"""

    if isinstance(variables, flax.core.FrozenDict):
        variables = variables.unfreeze()

    if not isinstance(variables, dict):
        return []

    out = []
    for k in variables.keys():
        if isinstance(variables[k], dict):
            for suffix in var_list(variables[k]):
                out.append(k + '.' + suffix)
        else:
            out.append(k)
    return out


def var_replace(variables, vid, value):
    is_frozen = False
    if isinstance(variables, flax.core.FrozenDict):
        is_frozen = True
        variables = variables.unfreeze()

    variables = _var_retrieve(variables, vid, value)

    if is_frozen:
        variables = flax.core.FrozenDict(variables)

    return variables


class Model(nn.Module):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *input_args, variables=None, method_name=None, rngs=None):
        if variables is None:
            variables = {}

        if method_name is None:
            ret = self.apply(variables, *input_args, rngs)
        elif isinstance(method_name, str):
            ret = self.apply(variables, *input_args, method=lambda module, *i: _rgetattr(module, method_name)(*i), rngs=rngs)
        else:
            ret = self.apply(variables, *input_args, method=method_name, rngs=rngs)
        return ret

    def var(self, name, mode='update', init_fn=None, shape=None, dtype=None):
        """Wrapper function for param and variable"""
        # updateable/trainable, non-trainable but assignable parameter
        if mode == 'update' or mode == 'u':
            v = self.param(name, init_fn, shape, dtype)
        elif mode == 'fixed' or mode == 'f':
            v = self.variable('fixed', name, init_fn, shape, dtype)
        else:
            raise ValueError('mode of var must be update/u or fixed/f.')

        return v

    def log_intermediate(self):
        raise NotImplementedError

    def var_find(self, variables, s):
        """Find a variable by keyword matching and return its unique identifier."""

        def recursive_lookup(k, d):
            if k in d: return d[k]
            for v in d.values():
                if isinstance(v, dict):
                    a = recursive_lookup(k, v)
                    if a is not None: return a
            return None

        raise NotImplementedError

    def var_verify(self, variables):
        """Verify if the given variables is compatiable with the current model."""
        # probably remove this. the implementation has to go through apply function. then why not just try apply.
        raise NotImplementedError

    def model_hyperparams(self):
        """serialize all model parameters, return json etc."""
        raise NotImplementedError

    @classmethod
    def recover_from_hyperparams(cls, s):
        raise NotImplementedError

    def model_save(self):
        raise NotImplementedError