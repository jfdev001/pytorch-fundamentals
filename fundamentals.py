"""Script for PyTorch Model of Random Data"""

from multiprocessing.sharedctypes import Value
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class RandomNormalDataset(Dataset):
    """Dataset for random normal data."""

    def __init__(
            self,
            samples: int = 128,
            x_dims: int = 4,
            y_dims: int = 1,
            seed: int = 0,
            regression: bool = True,
            onehot: bool = False):
        """Define state for RandomNormalDataset

        Args:
            samples: Number of samples
            x_dims: Dimensions for inputs.
            y_dims: Dimensions for labels.
            seed: Random seed.
            regression: True for real labels, else integer labels.
            onehot: True for one hotting labels, false otherwise.
        """

        # Save args
        self.samples = samples

        # Set seed
        np.random.seed(seed)

        # Define real inputs
        self.inputs = torch.normal(size=(samples, x_dims))

        # Define labels
        if regression:
            self.labels = torch.normal(size=(samples, y_dims))
        else:
            self.labels = torch.randint(low=0, high=y_dims, size=(samples,))
            if onehot:
                self.labels = F.one_hot(self.labels, num_classes=y_dims)

    def __len__(self):
        return self.samples

    def __get_item(self, idx: int):
        input_at_idx = self.inputs[idx]
        label_at_idx = self.labels[idx]
        return input_at_idx, label_at_idx


def main():
    pass


if __name__ == '__main__':
    main()
