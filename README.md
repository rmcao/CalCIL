# Berkeley Computational Imaging Library (CalCIL)
TL;DR: Write image reconstruction completely on a Jupyter notebook, and the notebook remains pleasant to open and read. 

## Describe the library 

## Installation
``$ pip3 install -e .``

## Pending essential functions:
- [x] Gradient-based iterative reconstruction
- [ ] Example notebooks
- [ ] forward.Model class implementation
  - [ ] Helper functions to read & search & manipulate variables dict
  - [ ] Easy access of intermediate variables
  - [ ] Save & restore a model (hyperparameters & code definition)
  - [ ] Easier way to declare a new variable
- [x] Register parameter dataclass
- [ ] Access of intermediate variables in loss fn and output fn
- [ ] Generic support for optax functionalities
- [ ] Refactor losses class 
- [ ] Multi-GPU support via pmap
- [ ] Populate unit testing
- [ ] Refactor output fn call or document the current API
- [ ] Automated hyperparameter storage, checkpoint - hyperparameter pairing
- [ ] Unambiguous interface for forward model. A clear interface with data loader
- [ ] Refactor data loader, integration with tf.dataset
- [ ] Migrate wave optics common functions
- [ ] Common signal processing functions (those missing in jax)

## Pending improvement features:
- [ ] napari integration
- [ ] Code formatting, linting
- [ ] Non-gradient descent optimization supports, e.g., linear model
- [ ] Store the forward model (incl. parameters, functions) in a file, recover the forward model without its definition code?
