import argparse
from online_lru.utils.util import str2bool
from online_lru.train import train
from online_lru.dataloading import Datasets
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="-1", help="which gpu to use")
    parser.add_argument("--USE_WANDB", type=str2bool, default=True, help="log with wandb?")
    parser.add_argument("--wandb_project", type=str, default="S5", help="wandb project name")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="ethz_joao",
        help="wandb entity name, e.g. username",
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        default="./cache_dir",
        help="name of directory where data is cached",
    )
    parser.add_argument(
        "--advanced_log",
        type=bool,
        default=False,
        help="Whether to aggregate more logging information (e.g. lambda trajs) accross training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=Datasets.keys(),
        default="mnist-classification",
        help="dataset name",
    )
    parser.add_argument(
        "--copy_pattern_length",
        type=int,
        default=10,
        help="which length for the copy task pattern to use",
    )
    parser.add_argument(
        "--copy_train_samples",
        type=int,
        default=100000,
        help="How many train samples to generate for the copy task",
    )
    parser.add_argument("--jax_seed", type=int, default=1919, help="seed randomness")

    # Model Parameters
    parser.add_argument(
        "--layer_cls",
        type=str,
        default="LRU",
        choices=["LRU", "RNN", "GRU"],
        help="What layer to use inside each block",
    )
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the network")
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Number of features, i.e. H, " "dimension of layer inputs/outputs",
    )
    parser.add_argument(
        "--d_hidden", type=int, default=256, help="Latent size of recurent unit, called H before"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="none",
        choices=["pool", "pool_st", "last", "none"],
        help="options: (for classification tasks) \\"
        "pool: mean pooling \\"
        "pool_st: cumulative mean pooling (with stop grad)"
        "last: take last element \\"
        "none: no pooling",
    )
    parser.add_argument(
        "--activation_fn",
        default="full_glu",
        type=str,
        choices=["full_glu", "half_glu1", "half_glu2", "gelu", "none"],
    )
    parser.add_argument(
        "--readout",
        default=0,
        type=int,
        help="Non zero to add a dense non linear layer after the encoder",
    )
    parser.add_argument(
        "--rnn_activation_fn",
        default="tanh",
        type=str,
        choices=["linear", "tanh", "relu"],
    )
    parser.add_argument(
        "--rnn_scaling_hidden",
        default=1.0,
        type=float,
        help="Additional normalization for th A matrix params to avoid explosion",
    )
    parser.add_argument(
        "--prenorm",
        type=str2bool,
        default=True,
        help="True: use prenorm, False: use postnorm",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="bptt",
        choices=[
            "bptt",
            "online_full",
            "online_full_rec",
            "online_full_rec_simpleB",
            "online_spatial",
            "online_1truncated",
            "online_snap1",
            "online_reservoir",
        ],
        help="training mode",
    )
    parser.add_argument("--r_min", type=float, default=0.0, help="r_min for LRU")

    parser.add_argument("--r_max", type=float, default=1.0, help="r_max for LRU")

    # Optimization Parameters
    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument(
        "--n_accumulation_steps",
        type=int,
        default=1,
        help="number of batches over which gradients are accumulated",
    )
    parser.add_argument("--epochs", type=int, default=100, help="max number of epochs")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=1000,
        help="number of epochs to continue training when val loss plateaus",
    )
    parser.add_argument("--lr_base", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=1,
        help="rec learning rate = lr_factor * lr_base",
    )
    parser.add_argument("--lr_min", type=float, default=0, help="minimum learning rate")
    parser.add_argument(
        "--cosine_anneal",
        type=str2bool,
        default=True,
        help="whether to use cosine annealing schedule",
    )
    parser.add_argument("--warmup_end", type=int, default=1, help="epoch to end linear warmup")
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1000000,
        help="patience before decaying learning rate for lr_decay_on_val_plateau",
    )
    parser.add_argument(
        "--reduce_factor",
        type=float,
        default=1.0,
        help="factor to decay learning rate for lr_decay_on_val_plateau",
    )
    parser.add_argument("--p_dropout", type=float, default=0.1, help="probability of dropout")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay value")
    parser.add_argument(
        "--opt_config",
        type=str,
        default="standard",
        choices=["standard", "Bdecay", "Bfast_and_decay", "C_slow_and_nodecay"],
        help="Opt configurations: \\ "
        "standard:          no weight decay on B (rec lr), weight decay on C (global lr) \\"
        "Bdecay:            weight decay on B (rec lr), weight decay on C (global lr) \\"
        "Bfast_and_decay:   weight decay on B (global lr), weight decay on C (global lr) \\"
        "Cslow_and_nodecay: no weight decay on B (rec lr), no weight decay on C (rec lr) \\",
    )

    # Logging parameters
    parser.add_argument(
        "--log_n_batches",
        type=int,
        default=20,
        help="number of batches per epoch to use to log gradient metrics",
    )
    parser.add_argument(
        "--log_loss_every",
        type=int,
        default=0,
        help="Log train loss every log_loss_every batches",
    )

    parser.add_argument(
        "--log_loss_at_time_steps",
        type=bool,
        default=False,
        help="Whether to log loss values at intermediate time steps",
    )

    parser.add_argument(
        "--enwik9_seq_len",
        type=int,
        default=1024,
        help="Sequence length of enwik9 task",
    )
    parser.add_argument(
        "--enwik9_train_samples",
        type=int,
        default=100000,
        help="How many train samples to generate for the enwik9 task",
    )

    # os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    train(parser.parse_args())
