# Description:
#  
# Written by Ruiming Cao on March 25, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from numbers import Number
import operator
import numpy as np
import jax.numpy as jnp
from jax.numpy import fft


# Convolution by fft implementation (fast for large conv kernel)
# from https://github.com/google/jax/pull/6343/commits/ce7c567384fd1b970386b7b9f5f4a6a8bbf1f4de
def fftconvolve(in1, in2, mode='full', axes=None):
    if in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.ndim == in2.ndim == 0:
        return in1 * in2
    elif in1.size == 0 or in2.size == 0:
        return jnp.array([], dtype=in1.dtype)
    in1, in2, axes = _standarize_freq_domain_conv_axes(in1, in2, mode, axes, sorted_axes=False)
    s1 = in1.shape
    s2 = in2.shape
    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]
    ret = _freq_domain_conv(in1, in2, axes, shape)
    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _freq_domain_conv(in1, in2, axes, shape):
    """Convolve `in1` with `in2` in the frequency domain."""
    if not len(axes):
        return in1 * in2
    in1_freq = fft.rfftn(in1, shape, axes=axes)
    in2_freq = fft.rfftn(in2, shape, axes=axes)
    ret = fft.irfftn(in1_freq * in2_freq, shape, axes=axes)
    return ret


def _standarize_freq_domain_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    """Handle the `axes` argument for `_freq_domain_conv`.
    Returns the inputs and axes in a standard form, eliminating redundant axes,
    swapping the inputs if necessary, and checking for various potential
    errors.
    """
    s1 = in1.shape
    s2 = in2.shape
    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)
    if not axes:
        raise ValueError("when provided, axes cannot be empty")
    # Axes of length 1 can rely on broadcasting rules for multipy, no fft needed.
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]
    if sorted_axes:
        axes.sort()
    if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
               for a in range(in1.ndim) if a not in axes):
        raise ValueError("incompatible shapes for in1 and in2:"
                         " {0} and {1}".format(s1, s2))
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        in1, in2 = in2, in1
    return in1, in2, axes


def _init_nd_shape_and_axes(x, shape, axes):
    """Handle shape and axes arguments for nd transforms"""
    if axes is not None:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]
        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")
    if shape is not None:
        shape = _iterable_of_int(shape, 'shape')
        if axes and len(axes) != len(shape):
            raise ValueError("when given, axes and shape arguments have to be of the same length")
        if axes is None:
            if len(shape) > x.ndim:
                raise ValueError("shape requires more axes than are present")
            axes = range(x.ndim - len(shape), x.ndim)
        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif axes is None:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]
    if any(s < 1 for s in shape):
        raise ValueError(
            "invalid number of data points ({}) specified".format(shape))
    return shape, axes


def _iterable_of_int(x, name=None):
    """Convert `x` to an sequence of ints"""
    if isinstance(x, Number):
        x = (operator.index(x),)
    try:
        x = [int(a) for a in x]
    except TypeError as e:
        name = name or 'value'
        raise ValueError("{} must be a scalar or iterable of integers"
                         .format(name)) from e
    return x


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Slice result based on the given `mode`."""
    if mode == 'full':
        return ret
    elif mode == 'same':
        return _centered(ret, s1)
    elif mode == 'valid':
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid)
    else:
        raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")


def _centered(arr, new_shape):
    """Centered slice of the given array."""
    new_shape = np.asarray(new_shape)
    start_idx = (arr.shape - new_shape) // 2
    end_idx = start_idx + new_shape
    centered_slice = tuple(slice(start_idx[k], end_idx[k]) for k in range(len(end_idx)))
    return arr[centered_slice]


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """True iff inputs need to be swapped to be compatible with 'valid' mode."""
    if mode != 'valid':
        return False
    if not shape1:
        return False
    if axes is None:
        axes = range(len(shape1))
    all_shape_1_gte_2 = all(shape1[i] >= shape2[i] for i in axes)
    all_shape_2_gte_1 = all(shape2[i] >= shape1[i] for i in axes)
    if not (all_shape_1_gte_2 or all_shape_2_gte_1):
        raise ValueError("For 'valid' mode, one array must be at least "
                         "as large as the other in every dimension")
    return not all_shape_1_gte_2
