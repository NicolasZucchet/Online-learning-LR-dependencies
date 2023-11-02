from functools import partial
from jax import random
import wandb
from .train_helpers import (
    create_train_state,
    reduce_lr_on_plateau,
    linear_warmup,
    cosine_annealing,
    constant_lr,
    train_epoch,
    validate,
)
from .dataloading import Datasets
from .seq_model import BatchClassificationModel
from .rec import init_layer


def train(args):
    """
    Main function to train over a certain number of epochs
    """
    best_test_loss = 100000000

    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(
            project=args.wandb_project,
            job_type="model_training",
            config=vars(args),
            entity=args.wandb_entity,
        )
    else:
        wandb.init(mode="offline")

    # Set rec learning rate lr as function of lr
    lr = args.lr_base
    rec_lr = args.lr_factor * lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Close over additional dataset-specific kwargs
    create_dataset_fn = Datasets[args.dataset]
    if args.dataset == "copy-pad-classification":
        create_dataset_fn = partial(
            create_dataset_fn,
            pattern_length=args.copy_pattern_length,
            train_samples=args.copy_train_samples,
        )
    elif args.dataset == "enwik9":
        create_dataset_fn = partial(
            create_dataset_fn,
            seq_len=args.enwik9_seq_len,
            train_samples=args.enwik9_train_samples,
        )

    # Dataset dependent logic
    if args.dataset in [
        "imdb-classification",
        "listops-classification",
        "aan-classification",
    ]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting training on `{args.dataset}` =>> Initializing...")

    # arguments specific to LRU or to RNN class
    additional_arguments = {}
    if args.layer_cls == "LRU":
        additional_arguments["r_min"] = args.r_min
        additional_arguments["r_max"] = args.r_max
    if args.layer_cls == "RNN":
        additional_arguments["activation"] = args.rnn_activation_fn
        additional_arguments["scaling_hidden"] = args.rnn_scaling_hidden
    elif args.layer_cls == "GRU":
        assert args.d_hidden == args.d_model
    rec_train = init_layer(
        layer_cls=args.layer_cls,
        d_hidden=args.d_hidden,
        d_model=args.d_model,
        seq_length=seq_len,
        training_mode=args.training_mode,
        **additional_arguments,
    )
    rec_val = init_layer(
        layer_cls=args.layer_cls,
        d_hidden=args.d_hidden,
        d_model=args.d_model,
        seq_length=seq_len,
        training_mode="bptt",  # bptt mode so rec does not keep in memory states and traces
        **additional_arguments,
    )

    model_cls = partial(
        BatchClassificationModel,
        rec_type=args.layer_cls,
        d_input=in_dim,
        d_output=n_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        seq_length=seq_len,
        padded=padded,
        activation=args.activation_fn,
        readout=args.readout,
        dropout=args.p_dropout,
        mode=args.mode,
        prenorm=args.prenorm,
        multidim=1 + (args.dataset == "copy-pad-classification"),
    )  # signature: (bool) training -> BatchClassificationModel
    model = model_cls(rec=rec_train, training_mode=args.training_mode, training=True)
    bptt_model = model_cls(rec=rec_val, training_mode="bptt", training=True)
    val_model = model_cls(rec=rec_val, training_mode="bptt", training=False)

    # initialize training state (optax) and internal states of the model
    state, init_states = create_train_state(
        model,
        init_rng,
        retrieval,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        opt_config=args.opt_config,
        rec_lr=rec_lr,
        lr=lr,
        n_accumulation_steps=args.n_accumulation_steps,
    )

    # Training Loop over epochs
    best_loss, best_accs, best_epoch = (100000000, {}, 0)  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count = 0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size / args.bsz)
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch + 1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch + 1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch + 1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (
            decay_function,
            rec_lr,
            lr,
            step,
            end_step,
            args.opt_config,
            args.lr_min,
        )
        train_rng, skey = random.split(train_rng)
        state, train_loss, step, init_states, metrics = train_epoch(
            state=state,
            init_states=init_states,
            rng=skey,
            model=model,
            bptt_model=bptt_model,
            trainloader=trainloader,
            seq_len=seq_len,
            in_dim=in_dim,
            lr_params=lr_params,
            log_n_batches=args.log_n_batches,
            log_loss_every=args.log_loss_every,
        )

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_accs, val_loss_time_dict = validate(
                state, val_model, valloader, seq_len, in_dim
            )
            val_loss_time_dict = {"Validation " + k: v for k, v in val_loss_time_dict.items()}

        print(f"[*] Running Epoch {epoch + 1} Test...")
        test_loss, test_accs, test_loss_time_dict = validate(
            state, val_model, testloader, seq_len, in_dim
        )
        test_loss_time_dict = {"Test " + k: v for k, v in test_loss_time_dict.items()}

        if valloader is None:
            val_loss, val_accs, val_loss_time_dict = test_loss, test_accs, test_loss_time_dict

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")

        print(
            f"Train Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} "
            f"-- Test Loss: {test_loss:.5f}"
        )
        for k in best_accs.keys():
            if valloader is not None:
                print(f"Val {k}: {val_accs[k]:.5f} ", end="")
            print(f"-- Test {k}: {test_accs[k]:.5f}")

        # For computing best epoch metrics and early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
            best_loss, best_accs, best_epoch = val_loss, val_accs, epoch
            if valloader is not None:
                best_test_loss, best_test_accs = test_loss, test_accs
            else:
                best_test_loss, best_test_accs = best_loss, best_accs
        else:
            count += 1

        # For learning rate decay purposes:
        input = lr, rec_lr, lr_count, val_loss, best_loss
        lr, rec_lr, lr_count = reduce_lr_on_plateau(
            input,
            factor=args.reduce_factor,
            patience=args.lr_patience,
            lr_min=args.lr_min,
        )

        # Print best accuracy & loss so far...
        print(f"> Best epoch: {best_epoch+1}")
        print(f"Best Val Loss: {best_loss:.5f} -- Best Test Loss: {best_test_loss:.5f}")
        for k in best_accs.keys():
            print(f"Best Val {k}: {best_accs[k]:.5f} -- Best Test {k}: {best_test_accs[k]:.5f}")

        accuracy_metrics = {}
        for k in val_accs.keys():
            accuracy_metrics["Val " + k] = val_accs[k]
            if valloader is not None:
                accuracy_metrics["Test " + k] = test_accs[k]
        wandb_metrics = {
            "Training Loss": train_loss,
            "Val loss": val_loss,
            "count": count,
            "Learning rate count": lr_count,
            "lr": state.opt_state.inner_opt_state.inner_states["regular"].inner_state.hyperparams[
                "learning_rate"
            ],
            "rec_lr": state.opt_state.inner_opt_state.inner_states["rec"].inner_state.hyperparams[
                "learning_rate"
            ],
            **accuracy_metrics,
            **metrics,
        }
        if valloader is not None:
            wandb_metrics["Test Loss"] = test_loss

        if args.log_loss_at_time_steps:
            wandb_metrics = {**wandb_metrics, **val_loss_time_dict, **test_loss_time_dict}

        wandb.log(wandb_metrics)

        wandb.run.summary["Best Val Loss"] = best_loss
        for k in best_accs.keys():
            wandb.run.summary["Best Val " + k] = best_accs[k]
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        for k in best_test_accs.keys():
            wandb.run.summary["Best Test " + k] = best_test_accs[k]

        if count > args.early_stop_patience:
            break
