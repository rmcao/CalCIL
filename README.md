# Berkeley Computational Imaging Library (CalCIL)

This project aims to provide a set of ubiquitous and reliable APIs for the general purpose computational imaging reconstructions. 

## Work in progress
This is still a messy construction yard. It's made public for the open source of [Speckle Flow SIM](https://github.com/Waller-Lab/SpeckleFlowSIM). Please open new issues or directly contact me (Ruiming Cao) for comments/suggestions.

## Installation
CaiCIL is built upon [Jax](https://github.com/google/jax) ecosystem. The core functionalities depend on Numpy, [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax). The logging in the reconstruction process requires Tensorflow V2.


Clone the repo

    $ git clone --recursive https://github.com/rmcao/CalCIL.git

Install CalCIL:

    $ cd calcil
    $ pip3 install -e .
