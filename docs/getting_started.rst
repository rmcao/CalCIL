.. _getting-started-ref-label:

Getting started
===============

This is a guide to getting started with CalCIL. We will walk you through the usage through a simple example. We assume that you have already installed CalCIL and its dependencies. If not, please refer to the `installation guide <installation-ref-label>`__.

Defining a convolution forward model
------------------------------------

A computational imaging challenge can be often written in the form of an inverse problem. For a linear imaging system, we have a linear operator :math:`A` and a vector :math:`x` that we want to recover from the measurement :math:`y`. The forward model can be written as:

.. math::
    y = A \cdot x

The most common forward model for optical imaging systems is the convolution with the given point spread function (PSF). We can define the forward model as a class in CalCIL.


.. code-block:: python

    from typing import Tuple
    import jax.numpy as jnp
    import calcil as cc

    class ConvImager(cc.forward.Model):
        dim_yx: Tuple[int, int]
        psf: jnp.ndarray

        def setup(self):
            # assume psf has the same shape as the unknown x
            assert self.dim_yx == self.psf.shape

            # prepare for convolution by FFT
            self.psf_pad = jnp.pad(self.psf, ((self.dim_yx[0] - self.dim_yx[0] // 2, self.dim_yx[0] // 2),
                                              (self.dim_yx[1] - self.dim_yx[1] // 2, self.dim_yx[1] // 2)))
            self.f_psf_pad = jnp.fft.rfft2(jnp.fft.ifftshift(self.psf_pad, axes=(-2, -1)), axes=(-2, -1))

        def __call__(self, x):
            """Forward model"""
            pad_x = jnp.pad(x, ((self.dim_yx[0] // 2, self.dim_yx[0] - self.dim_yx[0] // 2),
                                (self.dim_yx[1] // 2, self.dim_yx[1] - self.dim_yx[1] // 2)))
            out = jnp.fft.irfft2(jnp.fft.rfft2(pad_x, axes=(-2, -1)) * self.f_psf_pad, axes=(-2, -1))[self.dim_yx[0] // 2:-self.dim_yx[0] // 2, self.dim_yx[1] // 2:-self.dim_yx[1] // 2]

            return out

Here, we define a class `ConvImager` that inherits from `cc.forward.Model`. The `setup` method is used to prepare the PSF for convolution by padding and FFT. The `__call__` method is the forward model that takes an input `x` and returns the output `y`.

Using the forward model to simulate an imaging system
-----------------------------------------------------

Next, let's define a simple PSF and toy object :math:`x` and create an instance of the forward model to simulate the corresponding measurement :math:`y`.

.. code-block:: python

    psf = jnp.array([[0, 0, 0, 0, 0],
                     [0, 0, 0.3, 0, 0],
                     [0, 0.3, 1, 0.3, 0],
                     [0, 0, 0.3, 0, 0],
                     [0, 0, 0, 0, 0]])
    forward_model = ConvImager(dim_yx=(5, 5), psf=psf)

    x = jnp.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0]])

    # the empty dict is for the case when we have additional parameters specified by `self.params` in the model (see more below in the reconstruction)
    y = forward_model.apply({}, x=x)

    # visualize the input x, and output y
    import matplotlib.pyplot as plt

    f, axes = plt.subplots(1, 2)
    axes[0].imshow(x, cmap='gray')
    axes[0].set_title('x')
    axes[1].imshow(y, cmap='gray')
    axes[1].set_title('y')

The `apply` method is a wrapper around the `__call__` method. The empty dict allows for additional parameters to be passed to the forward model as specified by `self.params` (see more below in the reconstruction). In this case, we only need the input `x`.

Running deconvolution with gradient descent
-------------------------------------------

Now that we have the forward model defined, we can use it to perform deconvolution with gradient descent, assuming we don't already know the object :math:`x` but know the PSF used in forward model.

We first wrap around the forward model class and make it suitable for running reconstruction with calcil:

.. code-block:: python

    from flax import linen as nn

    class ConvImagerInv(cc.forward.Model):
        dim_yx: Tuple[int, int]
        psf: jnp.ndarray

        def setup(self):
            # Define the unknown x
            self.x = self.param('x', nn.initializers.zeros, self.dim_yx)

            # Use the forward model defined previously
            self.conv_imager = ConvImager(dim_yx=self.dim_yx, psf=self.psf)

        def __call__(self, input_dict):
            """forward model always has a input_dict input argument"""
            y = self.conv_imager(self.x)
            return y

    forward_model_inv = ConvImagerInv(dim_yx=(5, 5), psf=psf)

Preparing for data loading
^^^^^^^^^^^^^^^^^^^^^^^^^^

For gradient descent reconstruction, we define a dataloader which is a generator that yields a dictionary of input values each time.
In this case, each dictionary contains the same measurement $y$.
In general, we can use the built-in dataloader from `data_utils` module in CalCIL.

.. code-block:: python

    # prefix_dim is the shape of the batch dimension. In this case, it is (1,) since we have only one image.
    data_loader = cc.data_utils.loader_from_numpy({'y': y[jnp.newaxis]}, prefix_dim=(1,))

    print(next(data_loader))

Defining loss function
^^^^^^^^^^^^^^^^^^^^^^

Next, we define the loss function used for the update. Loss function is a callable that always takes `forward_output`, `variables`, `input_dict`, `intermediate` as arguments, and returns a scalar.
Then, `calcil` uses a `Loss` class to wrap around the loss function.

We define a simple L2 loss between the measurement `y` and the output of the forward model `y_hat` for the deconvolution problem.

.. code-block:: python

    # In this case, we use a pre-defined l2 loss function getter, which only requires the input dictionary key to retrieve the measurement from the input dictionary.
    # You may look into the source code of `get_l2_loss` to see how it is implemented.
    l2_loss = cc.loss.get_l2_loss('y')

    # register the loss function to calcil
    loss = cc.loss.Loss(l2_loss, 'l2')

Once we have the loss function defined, we need to register it to calcil using `cc.loss.Loss` wrapper.
The second argument is the name of the loss function, which will be useful for logging when there are multiple loss terms.

Setting up the initial values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to set up the initial value for the object $x$. The initial values are stored in a structured dictionary.

There are two ways to initialize such a dictionary:

* use the built-in `init` function.
* manually define the initial dictionary.

Here we show how to use the built-in `init` function:

.. code-block:: python

    import jax

    # using built-in init function to initialize the variables

    # random seed is needed to pass to the init function even though it won't be used for this case (no randomness)
    rng = jax.random.PRNGKey(0)
    variables = forward_model_inv.init(rng, input_dict=next(data_loader)[0])

    print(variables)

Alternatively, you can manually define the initial dictionary:

.. code-block:: python

    # manually define the initial dictionary
    variables = {'params': {'x': jnp.zeros((5, 5))}}

Setting up the optimization parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to set up the optimization parameters for the gradient descent algorithm.
Noteably, we need to specify the global parameters for the reconstruction and the learning parameters for the variables.
The global parameters include the number of epochs, the logging directory, the logging frequency, etc.
The learning parameters include the learning rate, the optimizer, and it is sometimes useful to specify different learning rates for different variables.

.. code-block:: python

    recon_param = cc.reconstruction.ReconIterParameters(save_dir='./checkpoint/demo_deconv', n_epoch=1000, log_every=10)

    var_params = cc.reconstruction.ReconVarParameters(lr=1e-1, opt='adam')

Full parameters can be found in the `calcil.reconstruction.ReconIterParameters` and `calcil.reconstruction.ReconVarParameters` classes in the `reconstruction` module.

Running the optimization
^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we can run the optimization using the `reconstruct_sgd` or `reconstruct_multivars_sgd` function in the `calcil.reconstruction` module.

.. code-block:: python

    recon, _ = cc.reconstruction.reconstruct_multivars_sgd(forward_model_inv.apply, variables, var_params,
                                                           data_loader, loss, recon_param)

    f, axes = plt.subplots(1, 2)
    axes[0].imshow(x, cmap='gray')
    axes[0].set_title('x')
    axes[1].imshow(recon['params']['x'], cmap='gray')
    axes[1].set_title('reconstructed x')

The full tutorial code can also be found in the `examples/demo_deconvolution.ipynb <https://github.com/rmcao/CalCIL/blob/master/examples/demo_deconvolution.ipynb>`__ in the CalCIL repository.