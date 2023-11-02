import jax
import jax.numpy as jnp
from flax import linen as nn
from .layers import SequenceLayer


class StackedEncoder(nn.Module):
    """
    Defines a stack of SequenceLayer to be used as an encoder.

    Args:
        rec             (nn.Module):    the recurrent module to use
        n_layers        (int32):        the number of SequenceLayer to stack
        dropout         (float32):      dropout rate
        d_input         (int32):        this is the feature size of the encoder inputs
        d_model         (int32):        this is the feature size of the layer inputs and outputs
                                        we usually refer to this size as H
        seq_length      (int32):        length of the time sequence considered
                                        we usually refer to this size as T
        activation      (string):       type of activation function to use
        training        (bool):         whether in training mode or not
        training_mode   (string):       type of training
        prenorm         (bool):         apply prenorm if true or postnorm if false
    """

    rec: nn.Module
    n_layers: int
    d_input: int
    d_model: int
    seq_length: int
    activation: str = "gelu"
    readout: int = 0
    dropout: float = 0.0
    training: bool = True
    training_mode: str = "bptt"
    prenorm: bool = False

    def setup(self):
        """
        Initializes a linear encoder and the stack of SequenceLayer.
        """
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                rec=self.rec,
                dropout=self.dropout,
                d_model=self.d_model,
                seq_length=self.seq_length,
                activation=self.activation,
                training=self.training,
                training_mode=self.training_mode,
                prenorm=self.prenorm,
            )
            for _ in range(self.n_layers)
        ]
        if self.readout > 0:
            self.mlp = nn.Dense(self.readout)

    def __call__(self, x):
        """
        Compute the TxH output of the stacked encoder given an Txd_input
        input sequence.
        Args:
             x (float32): input sequence (T, d_input)
        Returns:
            output sequence (float32): (T, d_model)
        """
        x = self.encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        if self.training_mode == "online_reservoir":
            x = jax.lax.stop_gradient(x)

        if self.readout > 0:
            x = nn.relu(self.mlp(x))
        return x

    def update_gradients(self, grad):
        # Update the gradients of encoder
        # NOTE no need to update other gradients as they will be updated through spatial backprop
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        for i, layer in enumerate(self.layers[::-1]):
            name_layer = "layers_%d" % (self.n_layers - i - 1)
            grad[name_layer] = layer.update_gradients(grad[name_layer])

        return grad


class ClassificationModel(nn.Module):
    """
    Classificaton sequence model. This consists of the stacked encoder, pooling across the
    sequence length, a linear decoder, and a softmax operation.

    Args:
        rec             (nn.Module):    the recurrent module to use
        n_layers        (int32):        the number of SequenceLayer to stack
        dropout         (float32):      dropout rate
        d_input         (int32):        this is the feature size of the encoder inputs
        d_model         (int32):        this is the feature size of the layer inputs and outputs
                                        we usually refer to this size as H
        d_output        (int32):        the output dimension, i.e. the number of classes
        padded:         (bool):         if true: padding was used
        mode            (str):          Options: [
                                            pool: use mean pooling,
                                            last: just take the last state,
                                            none: no pooling
                                        ]
        activation      (string):       type of activation function to use
        training        (bool):         whether in training mode or not
        training_mode   (string):       type of training
        prenorm         (bool):         apply prenorm if true or postnorm if false
        multidim        (int):          number of outputs (default: 1). Greater than 1 when
                                        several classificaitons are done per timestep
    """

    rec: nn.Module
    rec_type: str
    d_input: int
    d_output: int
    d_model: int
    n_layers: int
    seq_length: int
    padded: bool
    activation: str = "gelu"
    readout: int = 0
    dropout: float = 0.2
    training: bool = True
    training_mode: str = "bptt"
    mode: str = "pool"
    prenorm: bool = False
    multidim: int = 1

    def setup(self):
        """
        Initializes the stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoder(
            rec=self.rec,
            d_input=self.d_input,
            d_model=self.d_model,
            n_layers=self.n_layers,
            seq_length=self.seq_length,
            activation=self.activation,
            readout=self.readout,
            dropout=self.dropout,
            training=self.training,
            training_mode=self.training_mode,
            prenorm=self.prenorm,
        )
        self.decoder = nn.Dense(self.d_output * self.multidim)

    def decode(self, x, var=None):
        if var is None:
            x = self.decoder(x)
        else:
            x = self.decoder.apply(var, x)
        if self.multidim > 1:
            x = x.reshape(-1, self.d_output, self.multidim)
        return nn.log_softmax(x, axis=-1)

    def __call__(self, x):
        """
        Compute the size d_output log softmax output given a Txd_input input sequence.
        Args:
             x (float32): input sequence (T, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            # x, length = x  # input consists of data and prepadded seq lens
            x = x  # removed the length of the sequence for now

        x = self.encoder(x)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                raise (
                    NotImplementedError,
                    "removed the length from the inputs, doesn't work anymore",
                )
                # x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError(
                    "Mode must be in ['pool'] for self.padded=True (for now...)"
                )
            else:
                x = x[-1]
        elif self.mode in ["none"]:
            # Do not pool at all
            # if self.padded:
            #     raise NotImplementedError(
            #         "Mode must be in ['pool'] for self.padded=True (for now...)"
            #     )
            # else:
            # HACK: This includes padded parts of the input but proper masking requires some
            # rewriting
            x = x
        elif self.mode in ["pool_st"]:

            def cumulative_mean(x):
                return jnp.cumsum(x, axis=0) / jnp.arange(1, x.shape[0] + 1)[:, None]

            x = jax.lax.stop_gradient(cumulative_mean(x) - x) + x

        else:
            raise NotImplementedError("Mode must be in ['pool', 'pool_st', 'last', 'none']")

        return self.decode(x)

    def update_gradients(self, grad):
        # Update the gradients of encoder
        # NOTE no need to update other gradients as they will be updated through spatial backprop
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        grad["encoder"] = self.encoder.update_gradients(grad["encoder"])
        return grad


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=0,
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "traces": 0,
        "perturbations": 0,
    },
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False, "dropout": True},
)


def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = jnp.arange(L) < lengths
    return jnp.sum(mask[..., None] * x, axis=0) / lengths


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


# For Document matching task (e.g. AAN)
class RetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
    """

    d_model: int
    d_output: int

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        self.layer1 = nn.Dense(self.d_model)
        self.layer2 = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)
