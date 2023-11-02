import os
import unittest
from functools import partial
from online_lru.layers import SequenceLayer
from online_lru.rec import LRU
import jax
from utils import base_params, inputs, y, mask, compute_grads, check_grad_all
import flax.linen as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


batched_SequenceLayer = nn.vmap(
    SequenceLayer,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "traces": 0, "perturbations": 0},
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False, "dropout": True},
)


class TestLayers(unittest.TestCase):
    def test_online(self):
        """
        Check that the parameter update computed online by an SequenceLayer is
        the correct one, that is the gradient for all parameters that appear
        within or after the LRU layer.
        """
        lru = partial(LRU, training_mode="online_full", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.1,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_full",
            "training": True,
            "activation": "full_glu",
        }
        batched_seqlayer = batched_SequenceLayer(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        params_states = batched_seqlayer.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)},
            inputs,
        )
        _, online_grad = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_SequenceLayer(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "seq": [
                "B_re",
                "B_im",
                "C_re",
                "C_im",
                "D",
                "gamma_log",
                "nu",
                "theta",
            ],
            "out1": ["bias", "kernel"],
            "out2": ["bias", "kernel"],
        }
        check_grad_all(grad, online_grad, to_check=dict_check, atol=1e-1)
