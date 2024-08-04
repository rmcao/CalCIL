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
- The development of jax mainly relies on Google which may not be a good thing for some people

## Installation
```
# Create a virtual environment
conda create -n calcil python=3.9
conda activate calcil

# (optional, if needed) Install CUDA in conda virtual env
conda install -c conda-forge cudatoolkit~=11.8.0 cudnn~=8.8.0
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

# Install jaxlib for GPU
pip install jaxlib==0.3.18+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install this library
pip install git+https://github.com/rmcao/CalCIL.git
```
