# Description:
#  
# Written by Ruiming Cao on October 02, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import os
import datetime
import dataclasses
import functools
import time
from collections import defaultdict, abc
from typing import Callable, Union, Dict
import warnings

import flax.core
import numpy as np
import jax
from flax.metrics import tensorboard
from flax.training import train_state, checkpoints
import optax


@dataclasses.dataclass
class ReconIterParameters:
    save_dir: str
    n_iter: int
    keep_checkpoints: int = 1
    checkpoint_every: int = 10000
    save_imgs_every: int = 1000
    log_every: int = 100
    log_max_outputs: int = 5


@dataclasses.dataclass
class ReconVarParameters:
    lr_init: float = 0
    lr_decay_iter: int = 1e6
    lr_decay_rate: float = 1
    lr_decay_begin: int = 0
    delay_update_n_iter: int = 0
    optax_method: str = 'adam'
    update_every: int = 1
    grad_accumulate_iter: int = 1


def update_iter_sgd(state, input_dict, loss):
    (_, info), grad = jax.value_and_grad(loss, has_aux=True)(state.params, input_dict, state.apply_fn)
    new_state = state.apply_gradients(grads=grad)
    return new_state, info


def reconstruct_sgd(forward_fn: Callable,
                    variables: Union[Dict, flax.core.FrozenDict],
                    data_loader: abc.Generator,
                    loss_fn: Callable,
                    var_params: ReconVarParameters,
                    recon_param: ReconIterParameters,
                    output_fn: Union[Callable, None] = None,
                    post_update_handler: Callable = None):

    # define optimizer
    lr_schedule = optax.exponential_decay(init_value=var_params.lr_init,
                                          transition_steps=var_params.lr_decay_iter,
                                          transition_begin=var_params.lr_decay_begin,
                                          decay_rate=var_params.lr_decay_rate)
    if var_params.delay_update_n_iter > 0:
        warnings.warn('Delayed update for {} iterations (the optimization will do nothing).'.format(var_params.delay_update_n_iter))
        lr_schedule = optax.join_schedules([optax.constant_schedule(0.0), lr_schedule],
                                           [var_params.delay_update_n_iter])

    optimizer = getattr(optax, var_params.optax_method)(lr_schedule)

    if var_params.update_every > 1:
        optimizer = optax.chain(optax.apply_every(var_params.update_every), optimizer)
    if var_params.grad_accumulate_iter >= 2:
        optimizer = optax.MultiSteps(optimizer, var_params.grad_accumulate_iter)
    state = train_state.TrainState.create(apply_fn=forward_fn, params=variables, tx=optimizer)

    return run_reconstruction(state, data_loader, loss_fn, recon_param, output_fn, post_update_handler)


def generate_nested_dict_keys(d):
    if not isinstance(d, dict):
        return ['0']
    out = []
    for item in sorted(d.keys()):
        ret = generate_nested_dict_keys(d[item])
        for ret_item in ret:
            out.append(item + '_' + ret_item)
    return out


def reconstruct_multivars_sgd(forward_fn: Callable,
                              variables: Union[Dict, flax.core.FrozenDict],
                              var_params_pytree: Dict,
                              data_loader: abc.Generator,
                              loss_fn: Callable,
                              recon_param: ReconIterParameters,
                              output_fn: Union[Callable, None] = None,
                              post_update_handler: Callable = None):

    param_labels = generate_nested_dict_keys(var_params_pytree)

    list_var_params, params_treedef = jax.tree_util.tree_flatten(var_params_pytree)
    param_labels_pytree = jax.tree_util.tree_unflatten(params_treedef, param_labels)

    def _set_up_tx(var_params: ReconVarParameters):
        if var_params.lr_init <= 0:
            return optax.set_to_zero()

        lr_schedule = optax.exponential_decay(init_value=var_params.lr_init, transition_steps=var_params.lr_decay_iter,
                                              decay_rate=var_params.lr_decay_rate,
                                              transition_begin=var_params.lr_decay_begin)
        if var_params.delay_update_n_iter > 0:
            lr_schedule = optax.join_schedules([optax.constant_schedule(0.0), lr_schedule],
                                               [var_params.delay_update_n_iter])

        optimizer = getattr(optax, var_params.optax_method)(lr_schedule)
        if var_params.update_every > 1:
            optimizer = optax.chain(optax.apply_every(var_params.update_every), optimizer)
        if var_params.grad_accumulate_iter > 1:
            optimizer = optax.MultiSteps(optimizer, var_params.grad_accumulate_iter)
        return optimizer

    dict_opt = dict(zip(param_labels, [_set_up_tx(v_p) for v_p in list_var_params]))
    opt_multi = optax.multi_transform(dict_opt, param_labels_pytree)
    state = train_state.TrainState.create(
        apply_fn=forward_fn,
        params=variables.unfreeze() if isinstance(variables,flax.core.FrozenDict) else variables,
        tx=opt_multi)

    return run_reconstruction(state, data_loader, loss_fn, recon_param, output_fn, post_update_handler)


def run_reconstruction(state: train_state.TrainState,
                       data_loader: abc.Generator,
                       loss_fn: Callable,
                       recon_param: ReconIterParameters,
                       output_fn: Union[Callable, None] = None,
                       post_update_handler: Callable = None):
    # init logging
    summary_writer = tensorboard.SummaryWriter(os.path.join(recon_param.save_dir,
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    list_recon = defaultdict(list)

    # compile update function
    update_fn = jax.jit(functools.partial(update_iter_sgd, loss=loss_fn))
    post_update_fn = jax.jit(post_update_handler) if post_update_handler else None

    # set up timer
    loop_start_time = time.time()
    reset_timer = True

    # update model
    for s, input_dict in zip(range(recon_param.n_iter), data_loader):
        if reset_timer:
            loop_start_time = time.time()
            reset_timer = False

        state, info = update_fn(state, input_dict)

        if post_update_fn:
            state = post_update_fn(state)

        # print and log loss values
        if s % recon_param.log_every == 0:
            print(f'iter: {s}', end='')

            # to support both dataclass and dictionary as aux output obj
            if dataclasses.is_dataclass(info):
                for field in info.__dataclass_fields__:
                    value = getattr(info, field)
                    print(', {}: {:#.5g}'.format(field, value), end='')
                    summary_writer.scalar(field, value, s)
            else:
                for field in info:
                    value = info[field]
                    print(', {}: {:#.5g}'.format(field, value), end='')
                    summary_writer.scalar(field, value, s)

            reset_timer = True
            iter_per_sec = recon_param.log_every / (time.time() - loop_start_time)
            print(', iter per sec: {:#.5g}'.format(iter_per_sec))
            summary_writer.scalar('iter per sec', iter_per_sec, s)

        # save and log recon images
        if ((s > 0 and s % recon_param.save_imgs_every == 0) or
            (s == recon_param.n_iter - 1)) and (output_fn is not None):
            output_dict = output_fn(state)
            for key in output_dict:
                out_item = np.array(output_dict[key])
                list_recon[key].append(out_item)
                summary_writer.image(key, out_item, s, max_outputs=recon_param.log_max_outputs)

        # save checkpoints
        if (state.step % recon_param.checkpoint_every == 0) or (state.step == recon_param.n_iter):
            checkpoints.save_checkpoint(recon_param.save_dir, state, state.step,
                                        keep=recon_param.keep_checkpoints, overwrite=True)

    return state.params, list_recon


def load_checkpoint_and_output(load_path, output_fn=None):
    variables = checkpoints.restore_checkpoint(load_path, None)['params']
    if output_fn is None:
        return variables
    else:
        output_dict = output_fn(variables)
        return variables, output_dict
