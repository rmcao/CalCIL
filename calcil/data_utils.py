# Description: Data loaders
#  
# Written by Ruiming Cao on October 15, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import warnings
import numpy as np
import jax.tree_util


def loader_from_numpy(input_dict, prefix_dim=None, random=True, seed=951, aux_terms=None, nojax=True):
    """Load data from numpy arrays.

    Args:
        input_dict (dict): dictionary of numpy arrays. The first dimension is used as the batch dimension. All arrays
            should have the same length in the first dimension.
        prefix_dim (tuple): prefix dimension of the output arrays. It is used to control batch size in practice. If
            None, the first dimension is used.
        random (bool): whether to sample randomly. If False, the data is loaded in order.
        seed (int): random seed
        aux_terms (dict): auxiliary terms to be added to the output dictionary. The same terms are added to all
            batches.
        nojax (bool): whether to use jax or not
    Returns:
        generator: a generator that yields a dictionary of numpy arrays.
    """

    rng = np.random.default_rng(seed)

    if nojax:
        n_imgs = next(iter(input_dict.values())).shape[0]
    else:
        n_imgs = jax.tree_util.tree_leaves(input_dict)[0].shape[0]

    if prefix_dim:
        n_batches = int(np.floor(n_imgs / np.prod(prefix_dim)))
    else:
        n_batches = n_imgs

    print(f"n_imgs: {n_imgs}, n_batches: {n_batches}.")

    count_epoch, count_step = 0, 0

    while True:
        count_epoch += 1
        list_out_dict = []

        if random:
            indices = rng.permutation(n_imgs)
        else:
            indices = np.arange(n_imgs)

        for i_batch in range(n_batches):
            count_step += 1
            if prefix_dim is None:
                i = indices[i_batch]
                if nojax:
                    out_dict = {k: v[i] for k, v in input_dict.items()}
                else:
                    out_dict = jax.tree_util.tree_map(lambda x: x[i], input_dict)
            else:
                i = indices[i_batch * np.prod(prefix_dim):(i_batch+1) * np.prod(prefix_dim)]
                if nojax:
                    out_dict = {k: v[i].reshape(prefix_dim + v.shape[1:]) for k, v in input_dict.items()}
                else:
                    out_dict = jax.tree_util.tree_map(lambda x: x[i].reshape(prefix_dim + x.shape[1:]), input_dict)

            common_terms = {'epoch': count_epoch, 'step': count_step, 'batch': i_batch}
            if aux_terms:
                list_out_dict.append({**common_terms, **out_dict, **aux_terms})
            else:
                list_out_dict.append({**common_terms, **out_dict})

        yield list_out_dict


def tfds_files_loader(tf_dataset):

    raise NotImplementedError
