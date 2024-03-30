# CalCIL - gradient descent helper for computational image reconstruction via jax

This project aims to provide a set of ubiquitous and reliable APIs for the general purpose computational imaging reconstructions. 

## Features
- brainless gradient descent reconstruction 🤩
- fully customizable loss functions 🤓
- auto logging and visualization via tensorboard 🫡
- flexible optimization parameters (variable specific settings, learning rate schedule, etc.) 😬
- post-update custom functions 🤯
- handy helper functions for interactive 3/4D visualization on jupyter notebook 🔮

## Installation
CaiCIL is built upon [Jax](https://github.com/google/jax) ecosystem. The core functionalities depend on Numpy, [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax). The logging in the reconstruction process requires Tensorflow V2.

Clone the repo

    $ git clone --recursive https://github.com/rmcao/CalCIL.git

Install CalCIL:

    $ cd calcil
    $ pip3 install -e .


## History
This project serves as my personal "software infrastructure" during grad school. When I first started working on computational 
imaging in 2019, there are many different optimization algorithms used for imaging reconstruction, such as FISTA, ADMM, Gauss-Newton, conjugate gradient, etc.
**However, coming from a computer vision background, I really only know one thing: gradient descent.** 

It's not to say that gradient descent is a better algorithm to solve inverse problems, but it works well over a wide range of problems and is painless to implement using the existing deep learning frameworks.
All one has to do is to write out the forward model and the loss function, and the rest is taken care of by the deep learning framework, no more hand-derived gradient (I know this sounds like wayyyy too obvious now, but believe me it wasn't like this in 2015).

To brainlessly do gradient descent, I initially wrote a for-loop with tf auto-grad and copied & pasted that over and over again. 
Soon I realized that tf APIs are too cumbersome if all I want is just auto-grad (I was partially also annoyed that my old scripts won't work for tf v2). 
So I switched to jax and started to build this with common functions needed over multiple projects. 

