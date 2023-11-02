import jax.numpy as jnp
import jax
import numpy as np
import wandb


def cosine_similarity(array_a, array_b):
    dot_prod = jnp.sum(jnp.multiply(array_a, array_b))
    norm_a = jnp.linalg.norm(array_a)
    norm_b = jnp.linalg.norm(array_b)
    norm = norm_a * norm_b + 1e-8
    return dot_prod / norm


def relative_norm(array_a, array_b):
    norm_a = jnp.linalg.norm(array_a)
    norm_b = jnp.linalg.norm(array_b) + 1e-8
    return norm_a / norm_b


def concatenate_all(grad):
    flatten_grad = jax.tree_util.tree_map(lambda v: v.flatten(), grad)
    leaves_flatten_grad = jax.tree_util.tree_leaves(flatten_grad)
    return jnp.concatenate(leaves_flatten_grad)


def format_metrics(metrics):
    """
    Change name of keys for better grouping on weight and biases.
    """
    formatted_metrics = {}
    for key1 in sorted(metrics.keys()):
        for key2 in sorted(metrics[key1].keys()):
            formatted_metrics[f"{key1} / {key2}"] = metrics[key1][key2]
    return formatted_metrics


def merge_metrics(metrics_list):
    if isinstance(metrics_list[0], dict):
        all_keys = []
        for metric in metrics_list:
            all_keys += metric.keys()
        all_keys = list(np.unique(np.array(all_keys)))
        merged_dict = {}
        if any("param" in k for k in all_keys) or any(
            "grad" in k for k in all_keys
        ):  # HACK to check that we have "leave" dictionaries
            for metrics in metrics_list:
                merged_dict = {**merged_dict, **metrics}
            return merged_dict
        else:
            for key in all_keys:
                merged_dict[key] = merge_metrics(
                    [m[key] for m in metrics_list if (m is not None and key in m.keys())]
                )
        return merged_dict
    else:
        return metrics_list


def create_wandb_hist(values):
    hist = jnp.histogram(values.flatten(), bins=20)
    return wandb.Histogram(np_histogram=hist)


def get_last_key(pytree, last_key=None):
    if isinstance(pytree, dict):
        result = {}
        for key in pytree:
            result[key] = get_last_key(pytree[key], last_key=key)
        return result
    else:
        return last_key


def get_period_from_theta(theta):
    theta = jnp.mod(theta, 2 * jnp.pi)
    return 2 * jnp.pi / (jnp.pi - jnp.abs(theta - jnp.pi))


def get_grad_metrics(est_grad, true_grad):
    # Structure of grad is
    # encoder:
    #   encoder
    #   layers_0
    #   layers_1
    #   ...
    # decoder
    # Change the structure to encoder / layers_0 / layers_1 / ... / decoder for clearer logging
    est_grad = {**est_grad["encoder"], "decoder": est_grad["decoder"]}
    true_grad = {**true_grad["encoder"], "decoder": true_grad["decoder"]}

    # For all the level 0 dictionaries, concatenate all the leaves into one big vector
    for key in est_grad:
        est_grad[key]["all"] = concatenate_all(est_grad[key])
        true_grad[key]["all"] = concatenate_all(true_grad[key])
    est_grad["all"] = concatenate_all(est_grad)
    true_grad["all"] = concatenate_all(true_grad)

    # Norm of grad
    grad_norm = jax.tree_map(jnp.linalg.norm, est_grad)

    # Compute relative norm and cosine similarity between online grad and true grad
    rel_norm = jax.tree_map(relative_norm, est_grad, true_grad)
    cosine_sim = jax.tree_map(cosine_similarity, est_grad, true_grad)

    # Add metrics that are averages over all layers
    for metric in [cosine_sim, rel_norm, grad_norm]:
        if "layers_0" in metric.keys():
            average_layer = metric["layers_0"]["all"]
            i = 1
            while "layers_%d" % i in cosine_sim.keys():
                average_layer += metric["layers_%d" % i]["all"]
                i += 1
            average_layer /= i
            metric["average_layers"] = average_layer

    metrics = jax.tree_util.tree_map(
        lambda cs, rn, gn: {
            "grad_cosine_similarity": cs,
            "grad_relative_norm": rn,
            "grad_norm": gn,
        },
        cosine_sim,
        rel_norm,
        grad_norm,
    )

    return metrics


def get_params_metrics(params):
    # Change the structure of parameters similarly to what's done for the gradient
    flat_params = {**params["encoder"], "decoder": params["decoder"]}
    for key in flat_params:
        flat_params[key]["all"] = concatenate_all(flat_params[key])
    flat_params["all"] = concatenate_all(flat_params)

    # Add norm of the parameters to the metrics
    norm_params = jax.tree_util.tree_map(
        lambda x: {"param_norm": jnp.linalg.norm(x.flatten())}, flat_params
    )

    # LRU specific metrics
    last_key = get_last_key(flat_params)
    gamma_log_hist = jax.tree_util.tree_map(
        lambda k, v: {"param_hist": create_wandb_hist(v)} if k == "gamma_log" else {},
        last_key,
        flat_params,
    )
    nu_hist = jax.tree_util.tree_map(
        lambda k, v: {"param_hist": create_wandb_hist(jnp.exp(-jnp.exp(v)))} if k == "nu" else {},
        last_key,
        flat_params,
    )
    theta_hist = jax.tree_util.tree_map(
        lambda k, v: {"param_hist": create_wandb_hist(jnp.exp(v))} if k == "theta" else {},
        last_key,
        flat_params,
    )
    period_hist = jax.tree_util.tree_map(
        lambda k, v: {"param_hist_period": create_wandb_hist(get_period_from_theta(jnp.exp(v)))}
        if k == "theta"
        else {},
        last_key,
        flat_params,
    )
    log_period_hist = jax.tree_util.tree_map(
        lambda k, v: {
            "param_hist_log_period": create_wandb_hist(jnp.log(get_period_from_theta(jnp.exp(v))))
        }
        if k == "theta"
        else {},
        last_key,
        flat_params,
    )

    params_metrics = merge_metrics(
        [norm_params, gamma_log_hist, nu_hist, theta_hist, period_hist, log_period_hist]
    )

    return params_metrics
