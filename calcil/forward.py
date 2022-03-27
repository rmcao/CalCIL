# Description:
#  
# Written by Ruiming Cao on January 04, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import functools

import jax.numpy as jnp
import jax
from flax import linen as nn


def _rgetattr(obj, attr, *args):
    """Recursive getattr. from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class Model(nn.Module):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *input_args, variables=None, method_name=None):
        if variables is None:
            variables = {}

        if method_name is None:
            ret = self.apply(variables, *input_args)
        else:
            ret = self.apply(variables, *input_args, method=lambda module, *i: _rgetattr(module, method_name)(*i))
        return ret

    def forward_obj(self, variables, rngs=None):
        self.bind(variables, rngs)
        raise NotImplementedError

    def var_find(self, variables, str):
        """Find a variable by keyword matching and return its unique identifier."""
        raise NotImplementedError

    def var_replace(self, variables, vid, value):
        raise NotImplementedError

    def var_verify(self, variables):

        raise NotImplementedError

    def var_print(self):
        raise NotImplementedError

    def model_hyperparams(self):
        """serialize all model parameters, return json etc."""
        raise NotImplementedError

    @classmethod
    def recover_from_hyperparams(cls, s):
        raise NotImplementedError

    def model_save(self):
        raise NotImplementedError