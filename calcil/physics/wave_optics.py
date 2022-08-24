# Description:
#  
# Written by Ruiming Cao on February 02, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import jax.numpy as jnp
import math

import numpy as np

from .. import forward

jnp_complex_datatype = jnp.complex64


def genGrid(size, dx, flag_shift = False):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        size    - length of the array
        dx      - pixel size
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        kx      - 1D Fourier grid

    """
    xlin = (jnp.arange(size,dtype=jnp_complex_datatype) - size//2) * dx
    if flag_shift:
        xlin = jnp.roll(xlin, (size)//2)
    return xlin


def genGridNumpy(size, dx, flag_shift = False):
    """This function generates 1D Fourier grid."""
    xlin = (np.arange(size,dtype='complex128') - size//2) * dx
    if flag_shift:
        xlin = np.roll(xlin, (size)//2)
    return xlin


def genPupil(shape, pixel_size, NA, wavelength, NA_in = 0.0, fx_illu=0.0, fy_illu=0.0):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = np.fft.ifftshift(genGridNumpy(shape[1], 1 / pixel_size / shape[1])).real
    fylin        = np.fft.ifftshift(genGridNumpy(shape[0], 1 / pixel_size / shape[0])).real
    pupil_radius = NA/wavelength
    pupil        = np.asarray((fxlin[np.newaxis,:] - fx_illu)**2 + (fylin[:,np.newaxis] - fy_illu)**2 <= pupil_radius**2)
    if NA_in != 0.0:
        pupil[(fxlin[np.newaxis,:] - fx_illu)**2 + (fylin[:,np.newaxis] - fy_illu)**2 < pupil_radius**2] = 0.0
    return jnp.asarray(pupil)


def propKernel(dim_yx, prop_distances, pixel_size, wavelength, RI):
    fxlin        = jnp.array(jnp.fft.ifftshift(genGrid(dim_yx[1], 1 / pixel_size / dim_yx[1])))
    fylin        = jnp.array(jnp.fft.ifftshift(genGrid(dim_yx[0], 1 / pixel_size / dim_yx[0])))

    prop_kernel = jnp.exp(1.0j*2.0*jnp.pi*jnp.abs(prop_distances[:, jnp.newaxis, jnp.newaxis])*(
            (RI/wavelength)**2 - fxlin[jnp.newaxis, jnp.newaxis,:]**2 - fylin[jnp.newaxis, :,jnp.newaxis]**2)**0.5)

    prop_kernel = jnp.where(prop_distances[:, jnp.newaxis, jnp.newaxis] < 0, prop_kernel.conj(), prop_kernel)
    return prop_kernel


def propKernelNumpy(shape, pixel_size, wavelength, prop_distance, NA=None, RI=1.0, band_limited=True):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin = np.fft.ifftshift(genGrid(shape[1], 1 / pixel_size / shape[1]))
    fylin = np.fft.ifftshift(genGrid(shape[0], 1 / pixel_size / shape[0]))
    if band_limited:
        assert NA is not None, "need to provide numerical aperture of the system!"
        Pcrop = genPupil(shape, pixel_size, NA, wavelength)
    else:
        Pcrop = 1.0
    prop_kernel = Pcrop * np.exp(1j * 2.0 * np.pi * np.abs(prop_distance) * Pcrop * (
                (RI / wavelength) ** 2 - fxlin[np.newaxis, :] ** 2 - fylin[:, np.newaxis] ** 2) ** 0.5)
    prop_kernel = prop_kernel.conj() if prop_distance < 0 else prop_kernel
    return prop_kernel


def cart2Pol(x, y):
    rho          = (x * jnp.conj(x) + y * jnp.conj(y))**0.5
    theta        = jnp.arctan2(jnp.real(y), jnp.real(x)).astype(jnp_complex_datatype)
    return rho, theta


def zernikePolynomial(z_index, shape, pixel_size, NA, wavelength):
    fxlin             = genGrid(shape[1], 1 / pixel_size / shape[1], flag_shift = True)
    fylin             = genGrid(shape[0], 1 / pixel_size / shape[0], flag_shift = True)
    fxlin             = jnp.tile(fxlin[jnp.newaxis,:], [shape[0], 1])
    fylin             = jnp.tile(fylin[:, jnp.newaxis], [1, shape[1]])
    rho, theta        = cart2Pol(fxlin, fylin)
    rho[:, :]        /= NA/wavelength

    n = int(jnp.ceil((-3.0 + jnp.sqrt(9 + 8 * z_index)) / 2.0))
    m = 2 * z_index - n * (n + 2)
    normalization_coeff = jnp.sqrt(2 * (n + 1)) if abs(m) > 0 else jnp.sqrt(n + 1)
    azimuthal_function = jnp.sin(abs(m) * theta) if m < 0 else jnp.cos(abs(m) * theta)
    zernike_poly = jnp.zeros([shape[0], shape[1]], dtype=jnp_complex_datatype)
    for k in range((n - abs(m)) // 2 + 1):
        zernike_poly[:, :] += ((-1) ** k * math.factorial(n - k)) / (
                    math.factorial(k) * math.factorial(0.5 * (n + m) - k) * math.factorial(
                0.5 * (n - m) - k)) * rho ** (n - 2 * k)

    return normalization_coeff * zernike_poly * azimuthal_function


class AberrationPupil(forward.Model):

    def setup(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError