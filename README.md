# CalCIL - gradient descent helper for computational image reconstruction via jax


<p align="center">
    <a style="text-decoration:none !important;" href="https://zenodo.org/doi/10.5281/zenodo.12786082" alt="DOI"><img src="https://zenodo.org/badge/779045683.svg" /></a>
    <a style="text-decoration:none !important;" href="https://calcil.readthedocs.io/en/latest/index.html" alt="documentation"> <img src="https://img.shields.io/badge/API-docs-34B167" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/BSD-3-Clause" alt="License"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
</p>


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

Detailed installation instructions can be found [here](https://calcil.readthedocs.io/en/latest/installation.html).
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

## Tutorials
A step-by-step tutorial on how to use CalCIL for image reconstruction can be found [here](https://calcil.readthedocs.io/en/latest/getting_started.html). 
Also, check out the example notebook for [image deconvolution](examples/notebook-deconvolution.ipynb). 

## Usage

The following work is powered by CalCIL:
- [Dynamic speckle structured illumination microscopy](https://arxiv.org/pdf/2206.01397) 
- [Neural space-time model](https://www.biorxiv.org/content/10.1101/2024.01.16.575950), [code repo](https://github.com/rmcao/nstm)
- [Space-Time DiffuserCam Video Reconstruction](https://opg.optica.org/abstract.cfm?uri=3d-2022-JW5B.1)

## Citation
If you find this library useful, please consider citing the following papers:
```
@inproceedings{cao2022dynamic,
  title={Dynamic structured illumination microscopy with a neural space-time model},
  author={Cao, Ruiming and Liu, Fanglin Linda and Yeh, Li-Hao and Waller, Laura},
  booktitle={2022 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--12},
  year={2022},
  organization={IEEE}
}

@article{cao2024neural,
  title={Neural space-time model for dynamic scene recovery in multi-shot computational imaging systems},
  author={Cao, Ruiming and Divekar, Nikita and Nu{\~n}ez, James and Upadhyayula, Srigokul and Waller, Laura},
  journal={bioRxiv 2024.01.16.575950},
  pages={2024--01},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```