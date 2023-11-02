import jax
import jax.numpy as jnp
from chex import assert_trees_all_equal_shapes
from online_lru.train_helpers import compute_grad
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

base_params = {
    "d_hidden": 10,
    "d_model": 2,
    "seq_length": 100,
}

inputs = jax.random.normal(
    jax.random.PRNGKey(0), (2, base_params["seq_length"], base_params["d_model"])
)

y = jnp.ones((2, base_params["seq_length"]))

mask = jax.numpy.ones((2, 100))


def loss_pred(pred, label, mask=None):
    if mask is None:
        return 0.5 * jnp.mean((pred - label) ** 2), jnp.zeros_like(pred)
    else:
        # mean loss over output axis:
        loss = jnp.mean((pred - label) ** 2, -1)
        # mean over mask loss
        return 0.5 * jnp.sum(mask * loss), jnp.zeros_like(pred)


def check_grad_all(grad_1, grad_2, to_check=None, **kwargs):
    # Check that the size matches
    assert_trees_all_equal_shapes(grad_1, grad_2)

    # Create a list of paths to variables to check
    if to_check is not None:
        paths, _ = jax.tree_util.tree_flatten_with_path(to_check)
        paths_to_check = []
        for path in paths:
            paths_to_check.append(
                "/".join([a.key for a in path[0] if a.__class__ == jax.tree_util.DictKey])
                + "/"
                + path[1]
            )
    else:
        flatten_grad_1 = jax.tree_util.tree_flatten_with_path(grad_1)[0]
        paths_to_check = [
            "/".join([a.key for a in g1[0] if a.__class__ == jax.tree_util.DictKey])
            for g1 in flatten_grad_1
        ]

    # For all the parameters to check, verify that they are close to each other
    for path in paths_to_check:
        keys = path.split("/")
        val1, val2 = grad_1, grad_2
        for key in keys:
            val1 = val1[key]
            val2 = val2[key]
        assert jnp.allclose(val1, val2, **kwargs), "Mismatch at %s" % path


def compute_grads(model, params_states, inputs, y, mask):
    rng = jax.random.PRNGKey(0)
    # Compute grad online
    _, online_grad = compute_grad(params_states, rng, inputs, y, mask, model)

    model.training_mode = "bptt"
    _, grad = compute_grad(params_states, rng, inputs, y, mask, model)
    return grad, online_grad
