.. _history-ref-label:

History of this project
=======================

This project serves as `my <https://rmcao.net>`__' personal "software infrastructure" during grad school. When I first started working on computational
imaging in 2019, there are many different optimization algorithms used for imaging reconstruction, such as FISTA, ADMM, Gauss-Newton, conjugate gradient, etc.
**However, coming from a computer vision background, I really only know one thing: gradient descent.**

It's not to say that gradient descent is a better algorithm to solve inverse problems, but it's often "good-enough" for a range of problems and is painless to implement using the existing deep learning frameworks.
All one has to do is to write out the forward model and the loss function, and the rest is taken care of by the deep learning framework, no more hand-derived gradient (I know this sounds like wayyyy too obvious now, but believe me it wasn't like this in 2015).

To brainlessly do gradient descent, I initially wrote a for-loop with tf auto-grad and copied & pasted that over and over again.
Soon I realized that tf APIs are too cumbersome if all I want is just auto-grad (I was partially also annoyed that my old scripts won't work for tf v2).
So I switched to jax and started to build this with common functions needed over multiple projects.

