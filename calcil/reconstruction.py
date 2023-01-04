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

from calcil.loss import Loss


@dataclasses.dataclass
class ReconIterParameters:
    save_dir: str
    n_epoch: int
    keep_checkpoints: int = 1
    checkpoint_every: int = 10000
    output_every: int = 1000
    log_every: int = 100
    log_max_imgs: int = 5


@dataclasses.dataclass
class ReconVarParameters:
    lr: float = 0
    opt: Union[str, optax.GradientTransformation] = 'adam'
    opt_kwargs: dataclasses.field(default_factory=dict) = None
    schedule: Union[str, optax.Schedule] = 'constant_schedule'
    schedule_kwargs: dataclasses.field(default_factory=dict) = None
    delay_update_n_iter: int = 0
    update_every: int = 1


def update_iter_sgd(state, input_dict, rngs, loss_fn):
    (_, info), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params, input_dict,
                                                             jax.tree_util.Partial(state.apply_fn, rngs=rngs))
    new_state = state.apply_gradients(grads=grad)
    return new_state, info


def _set_up_tx(var_params: ReconVarParameters):
    if var_params.lr <= 0:
        return optax.set_to_zero()

    if var_params.schedule_kwargs is None:
        schedule_kwargs = {}
    else:
        schedule_kwargs = var_params.schedule_kwargs

    if var_params.opt_kwargs is None:
        opt_kwargs = {}
    else:
        opt_kwargs = var_params.opt_kwargs

    if isinstance(var_params.schedule, str):
        if var_params.schedule == 'constant' or var_params.schedule == 'constant_schedule':
            lr_schedule = optax.constant_schedule(var_params.lr)
        elif var_params.schedule == 'exponential' or var_params.schedule == 'exponential_decay':
            lr_schedule = optax.exponential_decay(var_params.lr, **schedule_kwargs)
        elif var_params.schedule == 'linear' or var_params.schedule == 'linear_schedule':
            lr_schedule = optax.linear_schedule(var_params.lr, **schedule_kwargs)
        else:
            lr_schedule = getattr(optax, var_params.schedule)(var_params.lr, **var_params.schedule_kwargs)
    elif isinstance(var_params.schedule, optax.Schedule):
        lr_schedule = var_params.schedule
    else:
        raise ValueError('Unsupported input type for the learning rate schedule.')

    if var_params.delay_update_n_iter > 0:
        warnings.warn('Delayed update for {} iterations (the optimization will do nothing).'.format(var_params.delay_update_n_iter))
        lr_schedule = optax.join_schedules([optax.constant_schedule(0.0), lr_schedule],
                                           [var_params.delay_update_n_iter])

    if isinstance(var_params.opt, str):
        optimizer = getattr(optax, var_params.opt)(learning_rate=lr_schedule, **opt_kwargs)
    elif isinstance(var_params.opt, optax.GradientTransformation):
        optimizer = var_params.opt
    else:
        raise ValueError('Unsupported input type for the optimizer.')

    if var_params.update_every > 1:
        optimizer = optax.chain(optax.apply_every(var_params.update_every), optimizer)

    return optimizer


def reconstruct_sgd(forward_fn: Callable,
                    variables: Union[Dict, flax.core.FrozenDict],
                    data_loader: abc.Generator,
                    loss: Loss,
                    var_params: ReconVarParameters,
                    recon_param: ReconIterParameters,
                    output_fn: Union[Callable, None] = None,
                    post_update_handler: Callable = None,
                    rngs: Union[Dict, None] = None,
                    output_info: bool = False):
    optimizer = _set_up_tx(var_params)
    state = train_state.TrainState.create(apply_fn=forward_fn, params=variables, tx=optimizer)

    if output_info:
        return run_reconstruction(state, data_loader, loss, recon_param, output_fn, post_update_handler, rngs)
    else:
        return run_reconstruction(state, data_loader, loss, recon_param, output_fn, post_update_handler, rngs)[:2]


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
                              loss: Loss,
                              recon_param: ReconIterParameters,
                              output_fn: Union[Callable, None] = None,
                              post_update_handler: Callable = None,
                              rngs: Union[Dict, None] = None,
                              output_info: bool = False):

    param_labels = generate_nested_dict_keys(var_params_pytree)

    list_var_params, params_treedef = jax.tree_util.tree_flatten(var_params_pytree)
    param_labels_pytree = jax.tree_util.tree_unflatten(params_treedef, param_labels)

    dict_opt = dict(zip(param_labels, [_set_up_tx(v_p) for v_p in list_var_params]))
    opt_multi = optax.multi_transform(dict_opt, param_labels_pytree)
    state = train_state.TrainState.create(
        apply_fn=forward_fn,
        params=variables.unfreeze() if isinstance(variables,flax.core.FrozenDict) else variables,
        tx=opt_multi)

    if output_info:
        return run_reconstruction(state, data_loader, loss, recon_param, output_fn, post_update_handler, rngs)
    else:
        return run_reconstruction(state, data_loader, loss, recon_param, output_fn, post_update_handler, rngs)[:2]


def run_reconstruction(state: train_state.TrainState,
                       data_loader: abc.Generator,
                       loss: Loss,
                       recon_param: ReconIterParameters,
                       output_fn: Union[Callable, None],
                       post_update_handler: Callable,
                       rngs: Union[Dict, None]):
    # init logging
    summary_writer = tensorboard.SummaryWriter(os.path.join(recon_param.save_dir,
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    list_recon, list_info = defaultdict(list), defaultdict(list)

    # compile update function
    update_fn = jax.jit(functools.partial(update_iter_sgd, loss_fn=loss.get_loss_fn()))
    post_update_fn = jax.jit(post_update_handler) if post_update_handler else None

    # set up timer
    loop_start_time = time.time()
    reset_timer = True

    # update model
    for s, input_batches in zip(range(recon_param.n_epoch), data_loader):
        list_info['epoch'].append(s + 1)
        if reset_timer:
            loop_start_time = time.time()
            reset_timer = False

        batch_info = []
        for i_batch, input_dict in enumerate(input_batches):
            cur_rngs = jax.tree_map(lambda rng: jax.random.split(rng)[0], rngs)
            rngs = jax.tree_map(lambda rng: jax.random.split(rng)[1], rngs)

            state, info = update_fn(state, input_dict, cur_rngs)

            # accumulate info var
            batch_info.append(info)

        if post_update_fn:
            state = post_update_fn(state)

        # print and log loss values
        if (s + 1) % recon_param.log_every == 0:
            print(f'epoch: {s + 1}', end='')

            if isinstance(batch_info[0], dict):
                info_avg = {}
                for field in batch_info[0].keys():
                    info_avg[field] = sum(float(info[field]) for info in batch_info) / len(batch_info)
                    print(', {}: {:#.5g}'.format(field, info_avg[field]), end='')
                    summary_writer.scalar(field, info_avg[field], s + 1)

                for field in info_avg:
                    list_info[field].append(info_avg[field])

            reset_timer = True
            epoch_per_sec = min(s + 1, recon_param.log_every) / (time.time() - loop_start_time)
            print(', epoch per sec: {:#.5g}'.format(epoch_per_sec))
            summary_writer.scalar('epoch per sec', epoch_per_sec, s + 1)
            list_info['elapsed time'].append(time.time() - loop_start_time)

        # save and log recon images
        if (((s + 1) % recon_param.output_every == 0) or
            ((s + 1) == recon_param.n_epoch)) and (output_fn is not None):
            output_dict = output_fn(state.params, state)
            for key in output_dict:
                out_item = np.array(output_dict[key])
                list_recon[key].append(out_item)
                summary_writer.image(key, out_item, s + 1, max_outputs=recon_param.log_max_imgs)

        # save checkpoints
        if (state.step % recon_param.checkpoint_every == 0) or (state.step == recon_param.n_epoch):
            checkpoints.save_checkpoint(recon_param.save_dir, state, state.step,
                                        keep=recon_param.keep_checkpoints, overwrite=True)

    return state.params, list_recon, list_info


def load_checkpoint_and_output(load_path, output_fn=None):
    variables = checkpoints.restore_checkpoint(load_path, None)['params']
    if output_fn is None:
        return variables
    else:
        output_dict = output_fn(variables)
        return variables, output_dict
