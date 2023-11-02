import os
from pathlib import Path

import numpy as np
import torch

DATAPATH = Path(os.path.expanduser("~/data/jax"))


def create_enwik9_dataloader(bsz: int, seq_len: int, train_samples: int, split: str, seed: int):
    """
    Creates a dataloader for the [enwik9 dataset](https://mattmahoney.net/dc/textdata.html).

    Assumes the following preprocessing has been done:
    ```
    import numpy as np

    f =  np.fromfile("enwik9", dtype=np.uint8)
    train_len = int(f.shape[0] * 0.9)
    valid_len = int(f.shape[0] * 0.05)
    print(train_len)
    np.save("train.npy", f[:train_len])
    np.save("valid.npy", f[train_len:train_len + valid_len])
    np.save("test.npy", f[train_len + valid_len:])
    ```

    Args:
        bsz (int): batch size
        seq_len (int): sequence length
        split (str): train, valid or test
        seed (int): random seed

    Returns:
        DataLoader: dataloader for enwik9
    """
    assert split in ["train", "valid", "test"]
    data = np.load(os.path.join(DATAPATH, "enwik9", "{}.npy".format(split)))

    # Reshape data into chunks of seq_len + 1 to create autoregressive targets
    data = data[: data.shape[0] - (data.shape[0] % (seq_len + 1))]  # trim off remainder
    data = data.reshape(-1, seq_len + 1)

    if train_samples is not None:
        assert train_samples <= data.shape[0]
        data = data[:train_samples]

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data))

    def collate_fn(batch):
        """Create time-lagged auto-regressive target"""
        x = torch.stack([b[0][:-1] for b in batch], dim=0)
        y = torch.stack([b[0][1:] for b in batch], dim=0)
        return (x, y)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
