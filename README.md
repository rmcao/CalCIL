# CalCIL - gradient descent helper for computational image reconstruction via jax

## Features
- 🧠 brainless gradient descent-based image reconstruction 
- 🤓 fully customizable loss functions 
- 🫡 auto logging and visualization via tensorboard
- 😬 flexible optimization parameters (variable specific settings, learning rate schedule, etc.)
- 🤯 post-update custom functions 
- 🔮 handy helper functions for interactive 3/4D visualization on jupyter notebook 

### Why using jax?

- jax.numpy as a drop-in replacement for numpy, and also jax.scipy for scipy
- Light weight, easy to debug, and [functional programming](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)
- [Slightly faster](https://www.kaggle.com/code/grez911/performance-of-jax-vs-pytorch/) [than pytorch](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/community-content/vertex_model_garden/benchmarking_reports/jax_vit_benchmarking_report.md)
- Good reproducibility through [explicit PRNG key management](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng)

### Why not using jax?

- Less commonly used than pytorch, so less community support
- Pretty barebone, often need other libraries from [jax ecosystem](https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/)

## Installation
The core functionalities depend on Numpy, [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax). The logging in the reconstruction process requires Tensorflow V2.

Clone the repo

    $ git clone --recursive https://github.com/rmcao/CalCIL.git

Create a virtual environment and install the dependencies

    $ conda create -n calcil python=3.10
    $ conda activate calcil

(Optional) Install CUDA in conda virtual env

    $ conda install -c conda-forge cudatoolkit~=11.8.0 cudnn~=8.8.0
    $ conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

Install jax

    $ pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.18+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl

Install CalCIL:

    $ cd calcil
    $ pip3 install -e .


## History
This project serves as my personal "software infrastructure" during grad school. When I first started working on computational 
imaging in 2019, there are many different optimization algorithms used for imaging reconstruction, such as FISTA, ADMM, Gauss-Newton, conjugate gradient, etc.
**However, coming from a computer vision background, I really only know one thing: gradient descent.** 

It's not to say that gradient descent is a better algorithm to solve inverse problems, but it's often "good-enough" for a range of problems and is painless to implement using the existing deep learning frameworks.
All one has to do is to write out the forward model and the loss function, and the rest is taken care of by the deep learning framework, no more hand-derived gradient (I know this sounds like wayyyy too obvious now, but believe me it wasn't like this in 2015).

To brainlessly do gradient descent, I initially wrote a for-loop with tf auto-grad and copied & pasted that over and over again. 
Soon I realized that tf APIs are too cumbersome if all I want is just auto-grad (I was partially also annoyed that my old scripts won't work for tf v2). 
So I switched to jax and started to build this with common functions needed over multiple projects. 

