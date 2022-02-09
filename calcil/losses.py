# Description: Implementation of common loss functions
#
# Written by Ruiming Cao on October 08, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io


def loss_l2_def(input_key: str, weight: float):

    def loss_l2(variables, input_dict, forward_fn):
        # forward_fn can be pre-defined as well? (like input_key, weight)

        # forward model
        I = forward_fn(variables, input_dict)

        # compute loss
        loss_l2 = ((input_dict[input_key] - I) ** 2).mean()
        loss = loss_l2 * weight

        aux = {"loss_l2": loss_l2}
        return loss, aux

    return loss_l2


def sum_losses(*losses):

    def loss_sum(variables, input_dict, forward_fn):
        loss_total = 0.0
        for loss in losses:
            # drop aux info for now
            l, _ = loss(variables, input_dict, forward_fn)
            loss_total = loss_total + l
        return loss_total, {}

    return loss_sum


def custom_loss(func):
    """Checker for custom loss function"""

    # check input parameters

    # check output

    # return the original func if pass all checks

    raise NotImplementedError