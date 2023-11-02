import math
from copy import deepcopy
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state
from jax.nn import one_hot
from tqdm import tqdm


from .log_helpers import format_metrics, get_grad_metrics, get_params_metrics, merge_metrics

"""
LR schedulers and optimizer related functions
"""


def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = jnp.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, rec_lr, count, new_loss, opt_loss = input
    if new_loss < opt_loss:
        count = 0
        opt_loss = new_loss
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        rec_lr = factor * rec_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if rec_lr < lr_min:
        rec_lr = lr_min

    return (
        lr,
        rec_lr,
        count,
    )


def constant_lr(step, base_lr, end_step, lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, rec_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    rec_lr_val = decay_function(step, rec_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_opt_state.inner_states["regular"].inner_state.hyperparams[
        "learning_rate"
    ] = jnp.array(lr_val, dtype=jnp.float32)
    state.opt_state.inner_opt_state.inner_states["rec"].inner_state.hyperparams[
        "learning_rate"
    ] = jnp.array(rec_lr_val, dtype=jnp.float32)
    if opt_config in ["Bdecay"]:
        # In this case we are applying the rec learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_opt_state.inner_states["none"].inner_state.hyperparams[
            "learning_rate"
        ] = jnp.array(rec_lr_val, dtype=jnp.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}

    return map_fn


def create_train_state(
    model,
    rng,
    retrieval,
    in_dim=1,
    bsz=128,
    seq_len=784,
    weight_decay=0.01,
    padded=False,
    opt_config="standard",
    rec_lr=1e-3,
    lr=1e-3,
    n_accumulation_steps=1,
):
    """
    Initializes the training state using optax
    """

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (jnp.ones((2 * bsz, seq_len, in_dim)), jnp.ones(2 * bsz))
        else:
            # removed length from inputs as masking is taken care in loss
            # dummy_input = (jnp.ones((bsz, seq_len, in_dim)), jnp.ones(bsz))
            dummy_input = jnp.ones((bsz, seq_len, in_dim))
    else:
        dummy_input = jnp.ones((bsz, seq_len, in_dim))

    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

    params = variables["params"]
    init_states = {
        "traces": variables.get("traces"),
        "perturbations": variables.get("perturbations"),
    }

    A_params = ["theta", "nu", "A", "norm"]  # NOTE: no weight decay on norm params
    B_params = ["B_re", "B_im", "B"]
    C_params = ["C_re", "C_im", "C"]
    gamma_params = ["gamma_log"]
    # By default:
    #   A, B, gamma (rec): rec learning rate and no weight decay
    #   C: global learning rate and weight decay
    if opt_config in ["standard"]:
        print("configuring standard optimization setup")
        rec_fn = map_nested_fn(
            lambda k, _: "rec"
            if k in A_params + B_params + gamma_params
            else ("none" if k in [] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "rec": optax.inject_hyperparams(optax.adam)(learning_rate=rec_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(
                    learning_rate=lr, weight_decay=weight_decay
                ),
            },
            rec_fn,
        )
    elif opt_config in ["Bdecay"]:
        # Apply weight decay to B (still with rec learning rate)
        print("configuring optimization with B in AdamW setup")
        rec_fn = map_nested_fn(
            lambda k, _: "rec"
            if k in A_params + gamma_params
            else ("none" if k in B_params else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(
                    learning_rate=rec_lr, weight_decay=weight_decay
                ),
                "rec": optax.inject_hyperparams(optax.adam)(learning_rate=rec_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(
                    learning_rate=lr, weight_decay=weight_decay
                ),
            },
            rec_fn,
        )

    elif opt_config in ["Bfast_and_decay"]:
        # Apply weight decay and global lr to B
        print("configuring optimization with B in AdamW setup with lr")
        rec_fn = map_nested_fn(
            lambda k, _: "rec"
            if k in A_params + gamma_params
            else ("none" if k in [] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "rec": optax.inject_hyperparams(optax.adam)(learning_rate=rec_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(
                    learning_rate=lr, weight_decay=weight_decay
                ),
            },
            rec_fn,
        )
    elif opt_config in ["Cslow_and_nodecay"]:
        # No weight decay for C and rec learning rate
        print("configuring optimization with C not in AdamW setup")
        rec_fn = map_nested_fn(
            lambda k, _: "rec"
            if k in A_params + B_params + C_params + gamma_params
            else ("none" if k in [] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "rec": optax.inject_hyperparams(optax.adam)(learning_rate=rec_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(
                    learning_rate=lr, weight_decay=weight_decay
                ),
            },
            rec_fn,
        )

    # Add steps aggragator
    tx = optax.MultiSteps(tx, n_accumulation_steps)

    def fn_is_complex(x):
        return x.dtype in [jnp.complex64, jnp.complex128]

    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(
        params
    )
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), init_states


"""
Loss functions and metrics
"""


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@jax.vmap
def create_mask(x, length):
    L = x.shape[0]
    mask = (jnp.arange(L) >= length[0]) * (jnp.arange(L) < length[1])
    return mask


def loss_fn(logits, labels, mask):
    """
    Pick the desired loss depending on the shape of the logits (and therefore the task)

    Args:
        logits (float32): network output (batch_size, seq_length(, dim_output), n_classes)
        labels (float32): label for each task (batch_size, seq_length(, dim_output))
        mask (float32): mask applied to the sequence to ignore some timesteps
            (batch_size, seq_length). Values taken: 0 or 1.

    NOTE: the seq_length dimension in labels comes from the need of online learning, as we want to
    have a learning signal for all time steps. We do not average over this dimension here because
    of chunking. Instead, we do it after computing the gradients.
    """
    # if seq_length dim is missing from the labels, we create it
    if len(logits.shape) == 3:
        # For SCIFAR, IMDB and ListOps
        losses = mask * cross_entropy_loss(logits, labels)
    elif len(logits.shape) == 4:
        # For copy task
        losses = mask * cross_entropy_loss(logits, labels).mean(axis=-1)
    else:
        raise NotImplementedError("Dataset incompatible with online learning")
    # losses has size (batch_size, seq_length)
    # Normalization by batch_size
    batch_size = logits.shape[0]
    losses = losses / batch_size
    # Normalization on sequence length as per mask size
    losses = batched_average_mask(losses, mask)
    return jnp.sum(losses), losses


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits, axis=-1) == label


def state_to_paramsstates(state, init_states, model):
    params_states = {"params": state.params}
    autodiff_all = model.training_mode in ["bptt", "online_spatial", "online_reservoir"]
    autodiff_seq = model.rec_type != "LRU" and model.training_mode == "online_1truncated"
    if not (autodiff_all or autodiff_seq):
        params_states["traces"] = jax.tree_util.tree_map(
            lambda s: jnp.zeros_like(s), init_states["traces"]
        )
        params_states["perturbations"] = jax.tree_util.tree_map(
            lambda s: jnp.zeros_like(s), init_states["perturbations"]
        )
    return params_states


@partial(jax.jit, static_argnums=(6, 7), donate_argnums=1)
def compute_grads(
    state,
    init_states,
    rng,
    inputs,
    labels,
    mask,
    model,
    bptt_model,
):
    params_states = state_to_paramsstates(state, init_states, model)
    _, true_grad = compute_grad(params_states, rng, inputs, labels, mask, bptt_model)
    if "online" in model.training_mode:
        _, est_grad = compute_grad(params_states, rng, inputs, labels, mask, model)
    else:
        est_grad = true_grad
    return init_states, true_grad, est_grad


"""
Data preprocessing
"""


def prep_batch(
    batch: tuple, seq_len: int, in_dim: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    inputs = jnp.asarray(inputs.numpy())
    # Grab lengths from aux if it is there.
    lengths = aux_data.get("lengths", None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = jnp.pad(inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(jnp.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = jnp.asarray(lengths.numpy()).astype(float)
        if len(lengths.shape) == 1:  # If lengths only give last
            lengths = jnp.stack([jnp.zeros((inputs.shape[0],)), lengths], axis=1)
        inputs = inputs.astype(float)
    else:
        inputs = inputs.astype(float)
        lengths = jnp.stack(
            [
                jnp.zeros(
                    inputs.shape[0],
                ),
                seq_len * jnp.ones((inputs.shape[0],)),
            ],
            axis=1,
        )
    # Convert and apply.
    targets = jnp.array(targets.numpy())

    if len(targets.shape) == 1:
        # Same label on the entire sequence if no label per output are there (we need it for online
        # learning)
        targets = jnp.repeat(jnp.expand_dims(targets, axis=-1), inputs.shape[-2], axis=-1)

    return inputs, targets.astype(float), lengths


"""
Update functions for a batch and for an entire epoch
"""


def train_epoch(
    state,
    init_states,
    rng,
    model,
    bptt_model,
    trainloader,
    seq_len,
    in_dim,
    lr_params,
    log_n_batches,
    log_loss_every,
):
    """
    Training function for an epoch that loops over batches.
    """
    batch_losses = []

    decay_function, rec_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for i, batch in enumerate(tqdm(trainloader)):
        inputs, labels, real_lengths = prep_batch(batch, seq_len, in_dim)
        mask = create_mask(inputs, real_lengths)
        rng, drop_rng = jax.random.split(rng)
        # Compute parameter metrics once every epoch
        if i == 0:
            params_metrics = get_params_metrics(deepcopy(state.params))
            # Aggregate gradient metrics over first log_n_batches batches
        if log_n_batches > 0 and i < log_n_batches:
            init_states, true_grad, est_grad = compute_grads(
                state,
                init_states,
                rng,
                inputs,
                labels,
                mask,
                model,
                bptt_model,
            )
            curr_grad_metrics = get_grad_metrics(est_grad, true_grad)
            if i == 0:
                grad_metrics = curr_grad_metrics
            else:
                grad_metrics = jax.tree_util.tree_map(
                    lambda old, new: (i * old + new) / (i + 1), grad_metrics, curr_grad_metrics
                )
        # Update the model
        state, loss, init_states = train_step(
            state, init_states, drop_rng, inputs, labels, mask, model
        )
        batch_losses.append(loss)
        lr_params = (decay_function, rec_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

        if log_loss_every > 0:
            if i % log_loss_every == 0:
                wandb.log({"train_loss": loss})

    # Merge and format metrics in a wandb friendly way
    if log_n_batches == 0:
        metrics = format_metrics(params_metrics)
    else:
        metrics = format_metrics(merge_metrics([params_metrics, grad_metrics]))

    # Return average loss over batches
    return (state, jnp.mean(jnp.array(batch_losses)), step, init_states, metrics)


def compute_accuracies(logits, labels, mask):
    if len(logits.shape) == 4:
        # Average over output dimensions and over active timesteps
        accs = {
            "Accuracy": jnp.sum(
                batched_average_mask(mask * compute_accuracy(logits, labels).mean(axis=-1), mask),
                axis=-1,
            )
        }
    elif len(logits.shape) == 3:
        indices = jnp.repeat(jnp.arange(logits.shape[1])[None, :], logits.shape[0], axis=0)
        last_index = jnp.max(indices * mask, axis=1)
        # Accuracy with last time-step prediction
        last_accs = compute_accuracy(logits[jnp.arange(logits.shape[0]), last_index], labels[:, 0])
        # Accuracy with mean logits prediction
        mean_preds = jnp.sum(mask[:, :, None] * logits, axis=1) / jnp.sum(mask, axis=1)[:, None]
        mean_accs = compute_accuracy(mean_preds, labels[:, 0])
        # Accuracy by ensembling predictions finally with ensembling predictions
        preds = jnp.argmax(logits, axis=-1)
        oh_pred = jax.nn.one_hot(preds, num_classes=2) * mask[:, :, None]
        ensemble_preds = jnp.sum(oh_pred, axis=1)
        ensemble_accs = compute_accuracy(ensemble_preds, labels[:, 0])
        accs = {
            "Accuracy (dense labels)": jnp.mean(compute_accuracy(logits, labels), axis=-1),
            "Accuracy (last token)": compute_accuracy(logits, labels)[:, -1],
            "Accuracy (last)": last_accs,
            "Accuracy (mean)": mean_accs,
            "Accuracy (ensemble)": ensemble_accs,
        }
    else:
        raise NotImplementedError("")
    return accs


def validate(state, model, testloader, seq_len, in_dim):
    """Validation function that loops over batches"""
    losses = []
    losses_unreduced = []
    accs = []
    for i, batch in enumerate(tqdm(testloader)):
        inputs, labels, real_lengths = prep_batch(batch, seq_len, in_dim)
        mask = create_mask(inputs, real_lengths)
        loss, acc, logits, loss_unreduced = eval_step(inputs, labels, mask, state, model)

        losses.append(loss)
        losses_unreduced.append(loss_unreduced)
        accs.append(acc)

    losses = jnp.stack(losses)
    losses_unreduced = jnp.concatenate(losses_unreduced)
    accs = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *accs)

    average_loss = jnp.mean(losses)
    losses_unreduced = jnp.mean(losses_unreduced, axis=0)  # Average over batch, not time
    average_accs = jax.tree_util.tree_map(lambda x: jnp.mean(x), accs)

    time_steps_to_log = [2**i for i in range(1, int(math.log2(seq_len)) + 1)]
    average_loss_per_time_step = {
        f"Loss (t={t})": losses_unreduced[t - 1] for t in time_steps_to_log
    }

    return average_loss, average_accs, average_loss_per_time_step


@jax.vmap
def batched_average_mask(a, mask):
    """Average of a by sum of values of mask"""
    return a / jnp.sum(mask)


def compute_grad(params_states, rng, inputs, labels, mask, model):
    def _loss_fn(ps):
        logits, mod_vars = model.apply(ps, inputs, rngs={"dropout": rng}, mutable=["traces"])
        loss = loss_fn(logits, labels, mask)[0]
        return loss, mod_vars

    (loss, mod_vars), grad = jax.value_and_grad(_loss_fn, has_aux=True)(params_states)
    autodiff_all = model.training_mode in ["bptt", "online_spatial", "online_reservoir"]
    autodiff_seq = model.rec_type != "LRU" and model.training_mode == "online_1truncated"
    if autodiff_all or autodiff_seq:
        grad = grad["params"]
    else:
        params_states = {
            "params": params_states["params"],
            "traces": mod_vars["traces"],
            "perturbations": grad["perturbations"],
        }
        grads_params = jax.tree_util.tree_map(
            lambda s: jnp.repeat(s[None, :], inputs.shape[0], axis=0) / inputs.shape[0],
            grad["params"],
        )
        grads = model.apply(params_states, grads_params, method=model.update_gradients)
        grad = jax.tree_util.tree_map(lambda s: jnp.sum(s, axis=0), grads)
    return loss, grad


@partial(jax.jit, static_argnums=6, donate_argnums=1)
def train_step(state, init_states, rng, inputs, labels, mask, model):
    """
    Performs a single training step given a batch of data
    NOTE: donate_argnums allows modifying the init_states in place, therefore avoiding copying the
    init_states twice
    """
    params_states = state_to_paramsstates(state, init_states, model)
    loss, grad = compute_grad(params_states, rng, inputs, labels, mask, model)
    state = state.apply_gradients(grads=grad)
    return state, loss, init_states


@partial(jax.jit, static_argnums=4)
def eval_step(inputs, labels, mask, state, model):
    logits = model.apply({"params": state.params}, inputs)
    # Loss is already normalized by batch size, we sum over to have the mean over the batch
    loss, loss_unreduced = loss_fn(logits, labels, mask)
    accs = compute_accuracies(logits, labels, mask)
    return loss, accs, logits, loss_unreduced


@partial(jax.jit, static_argnums=4)
def log_train_loss(inputs, labels, mask, state, model):
    logits = model.apply({"params": state.params}, inputs)
    _, losses = loss_fn(logits, labels, mask)
    losses = jnp.mean(losses, axis=0)  # average over batch_dim, leaving time dim
    time_steps_to_log = [2**i for i in range(1, int(math.log2(logits.shape[1])))]
    log_dict = {"train_loss_{}".format(t): losses[t] for t in time_steps_to_log}

    return log_dict
