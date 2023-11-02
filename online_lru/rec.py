from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from .rec_init import matrix_init, truncated_normal_matrix_init, theta_init, nu_init, gamma_log_init
from flax.core.frozen_dict import unfreeze


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def binary_operator_diag_spatial(q_i, q_j):
    """Same as above but stop the gradient for the recurrent connection"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, jax.lax.stop_gradient(A_j * b_i) + b_j


class LRU(nn.Module):
    """
    LRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    seq_length: int  # time sequence length
    gamma_norm: bool = True  # use gamma normalization
    exp_param: bool = True  # exponential parametrization for lambda
    r_min: float = 0.0  # smallest eigenvalue norm
    r_max: float = 1.0  # largest eigenvalue norm
    max_phase: float = 6.28  # max phase eigenvalue
    training_mode: str = "bptt"  # which learning algorithm that will be used
    training: bool = False  # TODO remove, for debugging purposes

    def get_diag_lambda(self, nu=None, theta=None):
        """
        Transform parameters nu and theta into the diagonal of the recurrent
        Lambda matrix.

        Args:
            nu, theta array[N]: when set to their default values, None, the
                parameters will take the values of the Module.
                NOTE: these arguments are added in order to backpropagate through this
                transformation.
        """
        if nu is None:
            nu = self.nu
        if theta is None:
            theta = self.theta
        if self.exp_param:
            theta = jnp.exp(theta)
            nu = jnp.exp(nu)
        return jnp.exp(-nu + 1j * theta)

    def get_diag_gamma(self):
        """
        Transform parameters gamma_log into the diagonal terms of the modulation matrix gamma.
        """
        if self.gamma_norm:
            return jnp.exp(self.gamma_log)
        else:
            return jnp.ones((self.d_hidden,))

    def get_B(self):
        """
        Get input to hidden matrix B.
        """
        return self.B_re + 1j * self.B_im

    def get_B_norm(self):
        """
        Get modulated input to hidden matrix gamma B.
        """
        return self.get_B() * jnp.expand_dims(self.get_diag_gamma(), axis=-1)

    def to_output(self, inputs, hidden_states):
        """
        Compute output given inputs and hidden states.

        Args:
            inputs array[T, H].
            hidden_states array[T, N].
        """
        C = self.C_re + 1j * self.C_im
        D = self.D
        y = jax.vmap(lambda x, u: (C @ x).real + D * u)(hidden_states, inputs)
        return y

    def get_hidden_states(self, inputs):
        """
        Compute the hidden states corresponding to inputs

        Return:
            hidden_states array[T, N]
        """
        # Materializing the diagonal of Lambda and projections
        diag_lambda = self.get_diag_lambda()
        B_norm = self.get_B_norm()

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        elements = (Lambda_elements, Bu_elements)
        if self.training_mode == "bptt":
            _, hidden_states = jax.lax.associative_scan(binary_operator_diag, elements)
        else:
            _, hidden_states = jax.lax.associative_scan(binary_operator_diag_spatial, elements)

        return hidden_states

    def setup(self):
        # Check that desired approximation is handled
        if self.training_mode == "online_snap1":
            raise NotImplementedError("SnAp-1 not implemented for LRU")
        assert self.training_mode in [
            "bptt",
            "online_full",
            "online_full_rec",
            "online_full_rec_simpleB",
            "online_snap1",  # same as online_full
            "online_spatial",
            "online_1truncated",
            "online_reservoir",
        ]
        self.online = "online" in self.training_mode  # whether we compute the gradient online
        if self.online:
            self.approximation_type = self.training_mode[7:]

        # NOTE if exp_param is true, self.theta and self.nu actually represent the log of nu and
        # theta lambda is initialized uniformly in complex plane
        self.theta = self.param(
            "theta",
            partial(theta_init, max_phase=self.max_phase, log=self.exp_param),
            (self.d_hidden,),
        )  # phase of lambda in [0, max_phase]
        self.nu = self.param(
            "nu",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max, log=self.exp_param),
            (self.d_hidden,),
        )  # norm of lambda in [r_min, r_max]
        if self.gamma_norm:
            self.gamma_log = self.param(
                "gamma_log", partial(gamma_log_init, log=self.exp_param), (self.nu, self.theta)
            )

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C_re = self.param(
            "C_re",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.C_im = self.param(
            "C_im",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))

        # Internal variables of the model needed for updating the gradient
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                partial(jnp.zeros, dtype=jnp.complex64),
                (self.seq_length, self.d_hidden),
            )

            self.traces_gamma = self.variable(
                "traces", "gamma", jnp.zeros, (self.seq_length, self.d_hidden)
            )
            self.traces_lambda = self.variable(
                "traces", "lambda", jnp.zeros, (self.seq_length, self.d_hidden)
            )

            if self.approximation_type in ["full", "snap1", "full_rec_simpleB"]:
                self.traces_B = self.variable(
                    "traces", "B", jnp.zeros, (self.seq_length, self.d_hidden, self.d_model)
                )

    def __call__(self, inputs):
        """
        Forward pass. If in training mode, additionally computes the eligibility traces that
        will be needed to compute the gradient estimate in backward.
        """
        # Compute hidden states and outputs
        hidden_states = self.get_hidden_states(inputs)
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            # To obtain the spatially backpropagated errors sent to hidden_states
            # NOTE: only works if pert_hidden_states is equal to 0
            hidden_states += self.pert_hidden_states.value
        output = self.to_output(inputs, hidden_states)

        # Compute and update traces if needed (i.e. if we are in online training mode)
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            Bu_elements = jax.vmap(lambda u: self.get_B() @ u)(inputs)
            # Update traces for B, lambda and gamma
            if self.approximation_type in ["1truncated"]:
                self.traces_lambda.value = hidden_states[:-1]
                self.traces_gamma.value = Bu_elements
            elif self.approximation_type in ["full", "full_rec", "full_rec_simpleB", "snap1"]:
                Lambda_elements = jnp.repeat(
                    self.get_diag_lambda()[None, ...], inputs.shape[0], axis=0
                )
                # Update for trace lambda
                _, self.traces_lambda.value = jax.lax.associative_scan(
                    binary_operator_diag,
                    (Lambda_elements[:-1], hidden_states[:-1]),
                )
                # Update for trace gamma
                Bu_elements_gamma = Bu_elements
                _, self.traces_gamma.value = jax.lax.associative_scan(
                    binary_operator_diag, (Lambda_elements, Bu_elements_gamma)
                )

            # Update trace for B
            if self.approximation_type in ["full", "snap1"]:
                full_Lambda_elements = jnp.repeat(
                    jnp.expand_dims(self.get_diag_lambda(), axis=-1)[None, ...],
                    inputs.shape[0],
                    axis=0,
                )  # same as left multiplying by diag(lambda), but same shape as B (to allow for
                #    element-wise multiplication in the associative scan)
                gammau_elements = jax.vmap(lambda u: jnp.outer(self.get_diag_gamma(), u))(
                    inputs
                ).astype(jnp.complex64)
                _, self.traces_B.value = jax.lax.associative_scan(
                    binary_operator_diag,
                    (full_Lambda_elements, gammau_elements + 0j),
                )
            elif self.approximation_type in ["full_rec_simpleB"]:
                self.traces_B.value = jax.vmap(lambda u: jnp.outer(self.get_diag_gamma(), u))(
                    inputs
                ).astype(jnp.complex64)
        return output

    def update_gradients(self, grad):
        """
        Eventually combine traces and perturbations to compute the (online) gradient.
        """
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        # We need to change the gradients for lambda, gamma and B
        # The others are automatically computed with spatial backpropagation
        # NOTE: self.pert_hidden_states contains dL/dhidden_states

        # Grads for lambda
        delta_lambda = jnp.sum(self.pert_hidden_states.value[1:] * self.traces_lambda.value, axis=0)
        _, dl = jax.vjp(
            lambda nu, theta: self.get_diag_lambda(nu=nu, theta=theta),
            self.nu,
            self.theta,
        )
        grad_nu, grad_theta = dl(delta_lambda)
        grad["nu"] = grad_nu
        grad["theta"] = grad_theta

        # Grads for gamma if needed
        if self.gamma_norm:
            delta_gamma = jnp.sum(
                (self.pert_hidden_states.value * self.traces_gamma.value).real, axis=0
            )
            # as dgamma/dgamma_log = exp(gamma_log) = gamma
            grad["gamma_log"] = delta_gamma * self.get_diag_gamma()

        # Grads for B
        if self.approximation_type in ["snap1", "full", "full_rec_simpleB"]:
            grad_B = jnp.sum(
                jax.vmap(lambda dx, trace: dx.reshape(-1, 1) * trace)(
                    self.pert_hidden_states.value, self.traces_B.value
                ),
                axis=0,
            )
            grad["B_re"] = grad_B.real
            grad["B_im"] = -grad_B.imag  # Comes from the use of Writtinger derivatives

        return grad


class RNN(nn.Module):
    """
    RNN layer that updates internal elegibility traces to allow online
    learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    seq_length: int  # time sequence length
    activation: str = "linear"  # activation function
    training_mode: str = "bptt"  # which learning algorithm that will be used
    scaling_hidden: float = 1.0  # additional scaling for the A matrix in the RNN

    def setup(self):
        # Check that desired approximation is handled
        assert self.training_mode in [
            "bptt",
            "online_spatial",
            "online_1truncated",
            "online_snap1",
        ]
        self.online = "online" in self.training_mode  # whether we compute the gradient online
        if self.online:
            self.approximation_type = self.training_mode[7:]
        else:
            self.approximation_type = "bptt"

        # Truncated normal to match haiku's initialization
        self.A = self.param(
            "A",
            partial(truncated_normal_matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_hidden, self.d_hidden),
        )
        self.B = self.param(
            "B",
            partial(matrix_init, normalization=jnp.sqrt(self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C = self.param(
            "C",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))
        if self.activation == "linear":
            self.act_fun = lambda x: x
        elif self.activation == "tanh":
            self.act_fun = jax.nn.tanh
        elif self.activation == "relu":
            self.act_fun = jax.nn.relu
        else:
            raise ValueError("Activation function not supported")

        # Internal variables of the model needed for updating the gradient
        if self.approximation_type in ["snap1"]:
            self.traces_A = self.variable(
                "traces", "A", jnp.zeros, (self.seq_length, self.d_hidden, self.d_hidden)
            )
            self.traces_B = self.variable(
                "traces", "B", jnp.zeros, (self.seq_length, self.d_hidden, self.d_model)
            )
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                partial(jnp.zeros, dtype=jnp.float32),
                (self.seq_length, self.d_hidden),
            )

    def get_hidden_states(self, inputs):
        """
        Compute the hidden states corresponding to inputs.

        Return:
            hidden_states array[T, N]
        """

        def _step(state, Bu):
            if self.training_mode in ["bptt"]:
                new_state = self.A @ self.act_fun(state) + Bu
            elif self.approximation_type in ["1truncated"]:
                new_state = self.A @ jax.lax.stop_gradient(self.act_fun(state)) + Bu
            else:
                new_state = jax.lax.stop_gradient(self.A @ self.act_fun(state)) + Bu
            return new_state, new_state

        Bu_elements = jax.vmap(lambda u: self.B @ u)(inputs)
        _, hidden_states = jax.lax.scan(_step, jnp.zeros(self.d_hidden), Bu_elements)

        return hidden_states

    def to_output(self, inputs, hidden_states):
        """
        Compute output given inputs and hidden states.

        Args:
            inputs array[T, H].
            hidden_states array[T, N].
        """
        return jax.vmap(lambda x, u: self.C @ x + self.D * u)(hidden_states, inputs)

    def __call__(self, inputs):
        """
        Forward pass. If in training mode, additionally computes the
        eligibility traces that will be needed to compute the gradient estimate
        in backward.
        """
        # Compute hidden states and output
        hidden_states = self.get_hidden_states(inputs)
        if self.approximation_type in ["snap1"]:
            # To obtain the spatially backpropagated errors sent to hidden_states
            hidden_states += self.pert_hidden_states.value
        output = self.to_output(inputs, hidden_states)

        # Update traces
        if self.approximation_type in ["snap1"]:
            # Repeat diagonal of A T times
            diags_A = jnp.repeat(jnp.diagonal(self.A)[None, ...], inputs.shape[0], axis=0)
            # Add the rho'(x) to it
            der_act_fun = jax.grad(self.act_fun)
            rho_primes = jax.vmap(lambda x: jax.vmap(der_act_fun)(x))(hidden_states)

            A_rho_prime_elements_N = jax.vmap(lambda x: jnp.outer(x, jnp.ones((self.d_hidden,))))(
                diags_A * rho_primes
            )
            A_rho_prime_elements_H = jax.vmap(lambda x: jnp.outer(x, jnp.ones((self.d_model,))))(
                diags_A * rho_primes
            )

            # Compute the trace of A
            # with tA_{t+1} = (diag A * rho'(x_t)) 1^T * tA_t + 1 rho(x_t)^T
            rho_x_elements = jax.vmap(lambda x: jnp.outer(jnp.ones((self.d_hidden,)), x))(
                self.act_fun(hidden_states)
            )
            _, self.traces_A.value = jax.lax.associative_scan(
                partial(binary_operator_diag),
                (A_rho_prime_elements_N, rho_x_elements),
            )

            # Compute the trace of B
            # with tB_{t+1} = (diag A * rho'(x_t)) 1^T * tB_t + 1 rho(x_t)^T
            u_elements = jax.vmap(lambda u: jnp.outer(jnp.ones((self.d_hidden,)), u))(inputs)
            _, self.traces_B.value = jax.lax.associative_scan(
                partial(binary_operator_diag), (A_rho_prime_elements_H, u_elements)
            )
        return output

    def update_gradients(self, grad):
        """
        Eventually combine traces and perturbations to compute the (online) gradient.
        """
        if self.approximation_type not in ["snap1"]:
            return grad

        # We need to change the gradients for A, and B
        grad["A"] = jnp.sum(
            jax.vmap(lambda dx, trace: dx.reshape(-1, 1) * trace)(
                self.pert_hidden_states.value[1:], self.traces_A.value[:-1]
            ),
            axis=0,
        )
        grad["B"] = jnp.sum(
            jax.vmap(lambda dx, trace: dx.reshape(-1, 1) * trace)(
                self.pert_hidden_states.value[1:], self.traces_B.value[:-1]
            ),
            axis=0,
        )
        return grad


class GRUOnlineCell(nn.recurrent.RNNCellBase, metaclass=nn.recurrent.RNNCellCompatibilityMeta):
    """
    GRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    training_mode: str
    activation_fn = jnp.tanh
    gate_fn = nn.sigmoid
    dtype = None

    def setup(self):
        dense_h = partial(
            nn.Dense,
            features=self.d_hidden,
            use_bias=False,
        )
        self.dense_hr = dense_h(name="hr")
        self.dense_hz = dense_h(name="hz")
        # add bias because the linear transformations aren't directly summed.
        self.dense_hn = dense_h(name="hn", use_bias=True)

        dense_i = partial(
            nn.Dense,
            features=self.d_hidden,
            dtype=self.dtype,
            use_bias=True,
        )
        self.dense_ir = dense_i(name="ir")
        self.dense_iz = dense_i(name="iz")
        self.dense_in = dense_i(name="in")

    def __call__(self, h, i):
        def sig_prime(x):
            return nn.sigmoid(x) * nn.sigmoid(1 - x)

        def tanh_prime(x):
            return 1 - jnp.tanh(x) ** 2

        if self.training_mode == "online_spatial":
            R = self.dense_ir(i) + jax.lax.stop_gradient(self.dense_hr(h))
        else:
            R = self.dense_ir(i) + self.dense_hr(h)
        r = nn.sigmoid(R)

        if self.training_mode == "online_spatial":
            Z = self.dense_iz(i) + jax.lax.stop_gradient(self.dense_hz(h))
        else:
            Z = self.dense_iz(i) + self.dense_hz(h)
        z = nn.sigmoid(Z)

        if self.training_mode == "online_spatial":
            hn = jax.lax.stop_gradient(self.dense_hn(h))
        else:
            hn = self.dense_hn(h)
        N = self.dense_in(i) + r * hn
        n = jnp.tanh(N)

        new_h = (1.0 - z) * h + z * n

        if self.training_mode in ["online_snap1"]:
            diag_dr_dh = sig_prime(R) * jnp.diag(self.dense_hr.variables["params"]["kernel"])
            diag_dz_dh = sig_prime(Z) * jnp.diag(self.dense_hz.variables["params"]["kernel"])
            diag_dn_dh = tanh_prime(N) * (
                diag_dr_dh * hn + r * jnp.diag(self.dense_hr.variables["params"]["kernel"])
            )
            diag_dh_dh = diag_dz_dh * (n - h) + diag_dn_dh * z + (1.0 - z)

            # Even though vscode tells you these are not used, they are.
            diag_dh_dz = (n - h) * sig_prime(Z)
            diag_dh_dn = z * tanh_prime(N)
            diag_dh_dr = z * tanh_prime(N) * hn * sig_prime(R)

            # Trick: use the diag of d.../dh as an error signal and then use vjp to compute
            # partial[i, i, :]
            grad = jax.tree_util.tree_map(jnp.zeros_like, unfreeze(self.variables["params"]))
            grad = unfreeze(grad)
            for beg in ["h", "i"]:
                for end in ["r", "z", "n"]:
                    dense = self.__getattr__("dense_" + beg + end)
                    inputs = eval(beg)
                    if beg == "h" and end == "n":
                        errors = errors = eval("diag_dh_d%s" % end) * r
                    else:
                        errors = eval("diag_dh_d%s" % end)

                    grad[beg + end] = jax.vjp(
                        lambda p: dense.apply({"params": p}, inputs),
                        unfreeze(dense.variables["params"]),
                    )[1](errors)[0]

            return new_h, (new_h, diag_dh_dh, {"params": grad})
        else:
            return new_h, new_h


class GRU(nn.Module):
    """
    GRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # NOT USE HERE input and output dimensions
    seq_length: int  # time sequence length
    training_mode: str = "bptt"  # which learning algorithm that will be used
    training: bool = False  # TODO remove, for debugging purposes
    activation_fn = jnp.tanh  # NOT USED
    dtype = None  # needed for obscure Flax reasons

    def setup(self):
        def update_bptt(cell, carry, inputs):
            return cell.apply({"params": cell.variables["params"]}, carry, inputs)

        def update_tbptt_and_spatial(cell, carry, inputs):
            return cell.apply(
                {"params": cell.variables["params"]}, jax.lax.stop_gradient(carry), inputs
            )

        def update_snap(cell, carry, inputs):
            hiddens, jac, t = carry
            new_hiddens, diag_rec_jac, partial_param = cell.apply(
                {"params": cell.variables["params"]}, hiddens, inputs
            )[1]
            new_hiddens += self.pert_hidden_states.value[t]

            def _jac_update(old, new):
                if len(old.shape) == 1:
                    diag = diag_rec_jac
                elif len(old.shape) == 2:
                    diag = diag_rec_jac[:, None]
                return new + diag * old

            new_jac = jax.tree_util.tree_map(_jac_update, jac, partial_param)

            return (new_hiddens, new_jac, t + 1), (new_hiddens, new_jac)

        self.cell = GRUOnlineCell(self.d_hidden, self.training_mode)
        if self.training_mode in ["bptt", "online_reservoir"]:
            update_fn = update_bptt
        elif self.training_mode in ["online_1truncated", "online_spatial"]:
            update_fn = update_tbptt_and_spatial
        elif self.training_mode == "online_snap1":
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                jnp.zeros,
                (self.seq_length, self.d_hidden),
            )
            update_fn = update_snap
        else:
            raise ValueError("Training mode not implemented:", self.training_mode)

        if self.is_initializing():
            _ = self.cell(jnp.zeros((self.d_hidden,)), jnp.zeros((self.d_hidden)))

        if self.training_mode == "online_snap1":
            # initialize all the traces
            traces_attr = []
            for e in jax.tree_util.tree_flatten_with_path(self.cell.variables["params"])[0]:
                path, val = e
                name_attr = "_".join([p.key for p in path])
                traces_attr += [name_attr]

                def _init(v):
                    return jnp.diagonal(
                        jnp.zeros(
                            [
                                self.seq_length,
                            ]
                            + [self.d_hidden] * len(v)
                        )
                    )

                val_attr = self.variable("traces", name_attr, _init, val.shape)
                self.__setattr__(name_attr, val_attr)
            self.traces_attr = traces_attr

        self.scan = nn.scan(
            update_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=self.seq_length,
        )

        self.update = update_bptt

    def __call__(self, inputs):
        if self.training_mode in [
            "bptt",
            "online_reservoir",
            "online_1truncated",
            "online_spatial",
        ]:
            init_carry = jnp.zeros((self.d_hidden,))
            return self.scan(self.cell, init_carry, inputs)[1]
        elif self.training_mode == "online_snap1":
            init_carry = (
                jnp.zeros((self.d_hidden,)),
                {
                    "params": jax.tree_util.tree_map(
                        lambda p: jnp.diagonal(jnp.zeros((self.d_hidden,) + p.shape)),
                        unfreeze(self.cell.variables["params"]),
                    )
                },
                0,
            )
            new_hiddens, traces = self.scan(self.cell, init_carry, inputs)[1]

            # Store the traces for gradient computation
            for e in jax.tree_util.tree_flatten_with_path(traces["params"])[0]:
                path, trace = e
                name_attr = "_".join([p.key for p in path])
                self.__getattr__(name_attr).value = trace

            return new_hiddens

    def update_gradients(self, grad):
        # Sanity check, gradient should be computed through autodiff for these methods
        if self.training_mode in [
            "bptt",
            "online_1truncated",
            "online_spatial",
            "online_reservoir",
        ]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        delta = self.pert_hidden_states.value

        # Otherwise grad[i] = delta[i] * trace[i]
        for attr in self.traces_attr:
            path = attr.split("_")
            trace = self.__getattr__(attr).value
            # for biases
            if len(trace.shape) == 2:
                g = delta * trace
            # for weights
            else:
                g = delta[:, None] * trace
            grad["cell"][path[0]][path[1]] = jnp.sum(g, axis=0)

        return grad


def init_layer(layer_cls, **kwargs):
    if layer_cls == "LRU":
        layer = LRU
    if layer_cls == "RNN":
        layer = RNN
    if layer_cls == "GRU":
        layer = GRU
    return partial(layer, **kwargs)
