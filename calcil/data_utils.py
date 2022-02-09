# Description: Data loaders
#  
# Written by Ruiming Cao on October 15, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import warnings
import numpy as np
import jax.tree_util


def simple_img_loader(input_dict, prefix_dim=None, consecutive_imgs=False, seed=951, aux_terms=None):
    # sample on the first dimension of each element in input_dict (assuming everything are ndarray)
    rng = np.random.default_rng(seed)
    n_imgs = jax.tree_util.tree_leaves(input_dict)[0].shape[0]
    count = 0

    while True:
        count = count + 1
        if prefix_dim is None:
            i = rng.integers(n_imgs)
            out_dict = jax.tree_util.tree_map(lambda x: x[i], input_dict)  # {'img': imgs[i], 't': ts[i]}
        else:
            if consecutive_imgs:
                i_begin = rng.integers(n_imgs)
                i = (np.arange(np.prod(prefix_dim)) + i_begin) % n_imgs
            else:
                i = rng.integers(n_imgs, size=np.prod(prefix_dim))
            out_dict = jax.tree_util.tree_map(lambda x: x[i].reshape(prefix_dim + x.shape[1:]), input_dict)

        if aux_terms:
            yield {'step': count, **out_dict, **aux_terms}
        else:
            yield {'step': count, **out_dict}


def epoch_img_loader(input_dict, prefix_dim, sequential=False, seed=81234, aux_terms=None):
    rng = np.random.default_rng(seed)
    n_imgs = jax.tree_util.tree_leaves(input_dict)[0].shape[0]
    batch_size = np.prod(prefix_dim)
    n_batch = int(n_imgs / batch_size)
    if n_batch * batch_size != n_imgs:
        warnings.warn('Some images cannot be reached in each iteration, n_imgs {}, n_batch {}'.format(n_imgs, n_batch))

    epoch_count = 0
    batch_count = 0
    while True:
        epoch_count = epoch_count + 1
        if sequential:
            i = np.arange(n_imgs)
        else:
            i = rng.permutation(n_imgs)

        for b in range(n_batch):
            batch_count += 1
            i_cur = i[b*batch_size:(b+1)*batch_size]
            out_dict = jax.tree_util.tree_map(lambda x: x[i_cur].reshape(prefix_dim + x.shape[1:]), input_dict)

            if aux_terms:
                yield {'step': batch_count, **out_dict, **aux_terms}
            else:
                yield {'step': batch_count, **out_dict}


def progressive_img_loader(input_dict, prefix_dim, incre_count, seed=951, aux_terms=None):
    rng = np.random.default_rng(seed)
    n_imgs = jax.tree_util.tree_leaves(input_dict)[0].shape[0]
    count = 0

    while True:
        r = min(count // incre_count + 1, n_imgs)
        count = count + 1
        i = rng.integers(r, size=np.prod(prefix_dim))
        out_dict = jax.tree_util.tree_map(lambda x: x[i].reshape(prefix_dim + x.shape[1:]), input_dict)

        if aux_terms:
            yield {'step': count, **out_dict, **aux_terms}
        else:
            yield {'step': count, **out_dict}


def balanced_progressive_img_loader(input_dict, prefix_dim, first_incre_count, alpha, seed=951, aux_terms=None):
    # number of appearance for each image is the same at the end.
    #
    assert alpha > 0.5, 'lam has to be greater than 0.5 for balanced loader'
    rng = np.random.default_rng(seed)
    n_imgs = jax.tree_util.tree_leaves(input_dict)[0].shape[0]
    count, idx_new_img = 0, 0
    incre_bound = first_incre_count
    cur_incre_count = first_incre_count

    while True:
        if count == incre_bound:
            idx_new_img = idx_new_img + 1
            cur_incre_count = int(cur_incre_count / (alpha + alpha / idx_new_img - 1 / idx_new_img))
            incre_bound = incre_bound + cur_incre_count

        if idx_new_img > n_imgs:
            i = rng.integers(n_imgs, size=np.prod(prefix_dim))
        else:
            mask_new_img = (rng.uniform(size=np.prod(prefix_dim)) < alpha).astype(int)
            i = rng.integers(max(1, idx_new_img), size=np.prod(prefix_dim))
            i = i * (1 - mask_new_img) + idx_new_img * np.ones_like(i) * mask_new_img

        count = count + 1
        out_dict = jax.tree_util.tree_map(lambda x: x[i].reshape(prefix_dim + x.shape[1:]), input_dict)

        if aux_terms:
            yield {'step': count, **out_dict, **aux_terms}
        else:
            yield {'step': count, **out_dict}
