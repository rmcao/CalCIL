.. _installation-ref-label:

Installation
============

Prerequisites
-------------

- NVIDIA GPU (GPU is not strictly required but highly recommended)
- `Anaconda <https://www.anaconda.com/products/individual>`__ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__

Step-by-step Installation
-------------------------

1. Create a conda virtual environment and activate it

   .. code-block:: bash

      $ conda create -n calcil python=3.10
      $ conda activate calcil

2. Install CalCIL. You may use -e flag to install in editable mode.

   .. code-block:: bash

      $ pip install git+https://github.com/rmcao/CalCIL.git

   .. note::

      This will install the standard CPU-only version of JAX by default.

3. (Optional) Enable GPU support.

   To enable GPU support, you must install the CUDA-enabled version of JAX. For example:

   .. code-block:: bash

      $ pip install -U "jax[cuda12]"

   Please refer to the official `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`__ for more details on installing JAX with CUDA or TPU support.

   .. note::

      To test the installation of jax, you can run the following command:

      .. code-block:: bash

         $ python -c "import jax.numpy as jnp; print(jnp.ones(5)+jnp.zeros(5))"

4. Install optional dependencies for interactive visualization via Jupyter lab

   .. code-block:: bash

      $ conda install -c conda-forge jupyterlab nodejs ipympl
