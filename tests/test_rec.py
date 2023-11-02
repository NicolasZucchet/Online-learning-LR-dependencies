import os
import unittest
import jax.numpy as jnp
from online_lru.rec import LRU, RNN
import jax
from tests.utils import base_params, inputs, y, mask, check_grad_all, compute_grads
import jax.lax
import flax.linen as nn
import flax

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Batched version of the LRU and RNN. We only do it as only the final model has to be batched in
# general
batched_LRU = nn.vmap(
    LRU,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "traces": 0, "perturbations": 0},
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False},
)
batched_RNN = nn.vmap(
    RNN,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "traces": 0, "perturbations": 0},
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False},
)


class TestLRU(unittest.TestCase):
    def test_toy(self):
        inputs = jnp.array([[[1.0], [2.0], [3.0]]])
        y = jnp.ones([1, 3])
        mask = jnp.ones([1, 3])
        # custom init of the params and state

        params_states = {
            "params": {
                "B_re": jnp.array([[2.0]]),
                "B_im": jnp.array([[2.0]]),
                "C_re": jnp.array([[2.0]]),
                "C_im": jnp.array([[1.0]]),
                "D": jnp.array([0.0]),
                "gamma_log": jnp.array([0.0]),
                "nu": jnp.log(jnp.log(jnp.array([2.0]))),
                "theta": jnp.array([-1e8]),
            },
            "traces": {
                "lambda": jnp.zeros([1, 3, 1], dtype=jnp.complex64),
                "gamma": jnp.zeros([1, 3, 1], dtype=jnp.complex64),
                "B": jnp.zeros([1, 3, 1, 1], dtype=jnp.complex64),
            },
            "perturbations": {
                "hidden_states": jnp.zeros([1, 3, 1], dtype=jnp.complex64),
            },
        }

        # Corresponding states
        # -- x hidden state
        # x_0 = 0
        # x_1 = 0.5 * 0 + 1 * (2 + 2j) * 1 = 2 + 2j
        # x_2 = 0.5 * (2 + 2j) + 1 * (2 + 2j) * 2 = 5 + 5j
        # x_3 = 0.5 * (5 + 5j) + 1 * (2 + 2j) * 3 = 8.5 + 8.5j
        # -- pred_y prediction
        # pred_y_1 = 2
        # pred_y_2 = 5
        # pred_y_3 = 8.5
        # -- delta_y BP error y
        # delta_y_1 = 1
        # delta_y_2 = 4
        # delta_y_3 = 7.5
        # -- delta_x BP error x
        # delta_x_1 = 1 * (2 + j) = 2 + j
        # delta_x_2 = 4 * (2 + j) = 8 + 4j
        # delta_x_3 = 7.5 * (2 + j) = 15 + 7.5j
        # delta_y BP error y
        # -- e_lambda
        # e_lambda_1 = 0.5 * 0 + 0 = 0
        # e_lambda_2 = 0.5 * 0 + 2 + 2j = 2 + 2j
        # e_lambda_3 = 0.5 * (2 + 2j) + 5 + 5j = 6 + 6j
        # -- dLambda
        # dLambda_1 = (2 + j) * 0 = 0
        # dLambda_2 = (8 + 4j) * (2 + 2j) = 8 + 24j
        # dLambda_3 = (15 + 7.5j) * (6 + 6j) = 45 + 135j
        # dLambda = 53 + 159j
        # -- e_gamma
        # e_gamma_1 = 0.5 * 0 + (2 + 2j) * 1 = 2 + 2j
        # e_gamma_2 = 0.5 * (2 + 2j) * 2 = 5 + 5j
        # e_gamma_3 = 0.5 * (5 + 5j) + (2 + 2j) * 3 = 8.5 + 8.5j
        # -- d_gamma
        # d_gamma_1 = Re[(2 + 2j) * (2 + j)] = 2
        # d_gamma_2 = Re[(5 + 5j) * (8 + 4j)] = 20
        # d_gamma_3 = Re[(8.5 + 8.5j) * (15 + 7.5j)] = 63.75
        # d_gamma = 85.75
        # -- e_B
        # e_B_1 = 0.5 * 0 + 1 = 1
        # e_B_2 = 0.5 * 1 + 2 = 2.5
        # e_B_3 = 0.5 * 2.5 + 3 = 4.25
        # -- dB
        # dB_1 = (2 - j) * 1 = 2 - j
        # dB_2 = (8 - 4j) * 2.5 = 20 - 10j
        # dB_3 = (15 - 7.5j) * 4.25 = 63.75 - 31.875j
        # dB = 85.75 - 42.875j

        # Compute gradient online
        batched_lru = batched_LRU(d_hidden=1, d_model=1, seq_length=3, training_mode="online_full")
        batched_LRU.rec_type = "LRU"

        grad, online_grad = compute_grads(batched_lru, params_states, inputs, y, mask)

        check_grad_all(grad, online_grad, atol=1e-5)

    def test_online_full(self):
        # Compute gradient online
        batched_lru = batched_LRU(**base_params, training_mode="online_full")
        batched_lru.rec_type = "LRU"
        params_states = batched_lru.init({"params": jax.random.PRNGKey(0)}, inputs)

        grad, online_grad = compute_grads(batched_lru, params_states, inputs, y, mask)

        # Check that the two match
        check_grad_all(grad, online_grad, atol=1e-2)

    def test_online_spatial(self):
        batched_lru = batched_LRU(**base_params, training_mode="online_spatial")
        batched_lru.rec_type = "LRU"
        def_params_states = batched_lru.init({"params": jax.random.PRNGKey(0)}, inputs)
        # Remove temporal recurrence
        params_states = {}
        params_states["params"] = flax.core.frozen_dict.unfreeze(def_params_states["params"])
        params_states["params"]["nu"] = jnp.ones_like(params_states["params"]["nu"]) * 1e8
        params_states["params"] = flax.core.frozen_dict.freeze(params_states["params"])

        grad, online_grad = compute_grads(batched_lru, params_states, inputs, y, mask)

        # Remove nu and theta from the comparison of the gradient and check that they are 0
        assert jnp.allclose(online_grad["nu"], jnp.zeros_like(online_grad["nu"]))
        assert jnp.allclose(online_grad["theta"], jnp.zeros_like(online_grad["theta"]))
        grad = {k: grad[k] for k in ["B_im", "B_re", "C_im", "C_re", "D", "gamma_log"]}
        online_grad = {
            k: online_grad[k] for k in ["B_im", "B_re", "C_im", "C_re", "D", "gamma_log"]
        }

        check_grad_all(online_grad, grad, atol=1e-3)


class TestRNN(unittest.TestCase):
    def test_online_snap1(self):
        # Compute gradient online
        batched_rnn = batched_RNN(**base_params, training_mode="online_snap1")
        batched_rnn.rec_type = "RNN"
        params_states = batched_rnn.init({"params": jax.random.PRNGKey(0)}, inputs)

        compute_grads(batched_rnn, params_states, inputs, y, mask)

    def test_online_spatial(self):
        batched_rnn = batched_RNN(**base_params, training_mode="online_spatial")
        batched_rnn.rec_type = "RNN"
        def_params_states = batched_rnn.init({"params": jax.random.PRNGKey(0)}, inputs)
        # Remove temporal recurrence
        params_states = {}
        params_states["params"] = flax.core.frozen_dict.unfreeze(def_params_states["params"])
        params_states["params"]["A"] = jnp.zeros_like(params_states["params"]["A"])
        params_states["params"] = flax.core.frozen_dict.freeze(params_states["params"])

        grad, online_grad = compute_grads(batched_rnn, params_states, inputs, y, mask)

        # Remove nu and theta from the comparison of the gradient and check that they are 0
        assert jnp.allclose(online_grad["A"], jnp.zeros_like(online_grad["A"]))
        assert jnp.allclose(online_grad["A"], jnp.zeros_like(online_grad["A"]))
        grad = {k: grad[k] for k in ["B", "C", "D"]}
        online_grad = {k: online_grad[k] for k in ["B", "C", "D"]}

        check_grad_all(online_grad, grad, atol=1e-3)

    def test_online_1truncated(self):
        # Compute gradient online
        batched_rnn = batched_RNN(**base_params, training_mode="online_1truncated")
        batched_rnn.rec_type = "RNN"
        params_states = batched_rnn.init({"params": jax.random.PRNGKey(0)}, inputs)

        compute_grads(batched_rnn, params_states, inputs, y, mask)
