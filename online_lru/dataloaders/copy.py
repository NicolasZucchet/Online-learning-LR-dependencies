from .base import SequenceDataset
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]


class CopyTask(SequenceDataset):
    _name_ = "copy"

    @property
    def init_defaults(self):
        return {
            "train_size": 100000,
            "val_size": 1000,
            "test_size": 1000,
            "in_dim": 8,
            "pattern_length": 10,
            "pad_length": 7,
        }

    def setup(self, stage=None):
        # TODO varying pattern length, autoregressive prediction and flip output
        n_total = self.train_size + self.val_size + self.test_size
        self.seq_length_in = self.pattern_length * 2 + self.pad_length + 1
        self.out_dim = self.in_dim - 1
        input_data = torch.zeros((n_total, self.seq_length_in, self.in_dim))
        output_data = torch.zeros((n_total, self.seq_length_in, self.in_dim - 1))

        pattern = 0.5 * torch.ones((n_total, self.pattern_length, self.in_dim - 1))
        pattern = torch.bernoulli(pattern)  # draw bernouillis with prob 0.5 of being 1
        # Add pattern to input and output
        input_data[:, : self.pattern_length, :-1] = pattern
        output_data[:, self.seq_length_in - self.pattern_length :, :] = pattern

        # Add stop token
        input_data[:, self.pattern_length + self.pad_length, -1] = 1

        # Create auxiliary data to give beginning and end of ouutput for mask
        lengths = torch.zeros((n_total, 2))
        lengths[:, 0] = self.seq_length_in - self.pattern_length
        lengths[:, 1] = self.seq_length_in

        self.dataset_train = ArrayDataset(
            input_data[: self.train_size],
            output_data[: self.train_size],
            lengths[: self.train_size],
        )
        self.dataset_val = ArrayDataset(
            input_data[self.train_size : self.train_size + self.val_size],
            output_data[self.train_size : self.train_size + self.val_size],
            lengths[self.train_size : self.train_size + self.val_size],
        )
        self.dataset_test = ArrayDataset(
            input_data[self.train_size + self.val_size :],
            output_data[self.train_size + self.val_size :],
            lengths[self.train_size + self.val_size :],
        )

        # Need to overwrite collate function such that we have the aux data for the mask.
        def collate_batch(batch):
            xs, ys, lengths = zip(*[(data[0], data[1], data[2]) for data in batch])
            xs = torch.stack(xs, dim=0)
            ys = torch.stack(ys, dim=0)
            lengths = torch.stack(lengths, dim=0)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch
