"""Script for PyTorch Model of Random Data"""

import argparse
from ast import parse
from distutils.util import strtobool

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
        # NOTE: Gradient requirements??? I think no
        # because think about this grad descent is
        # dC/dW or dC/dBias
        self.inputs = torch.rand(size=(samples, x_dims), requires_grad=False)

        # Define labels
        if regression:
            self.labels = torch.rand(
                size=(samples, y_dims), requires_grad=False)
        else:
            self.labels = torch.randint(
                low=0, high=y_dims, size=(samples,), requires_grad=False)
            if onehot:
                self.labels = F.one_hot(self.labels, num_classes=y_dims)

    def __len__(self):
        return self.samples

    def __get_item(self, idx: int):
        input_at_idx = self.inputs[idx]
        label_at_idx = self.labels[idx]
        return input_at_idx, label_at_idx


class MLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(self, ):
        """Define state for MLP."""
        pass

    def forward(self, x):
        """Forward pass."""
        pass


def cli(description: str):
    """Command line interface for fundamentals of pytorch."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--samples',
        type=int,
        default=128)
    parser.add_argument(
        '--x-dims',
        type=int,
        default=4)
    parser.add_argument(
        '--y-dims',
        type=int,
        default=1)
    parser.add_argument(
        '--seed',
        type=int,
        default=0)
    parser.add_argument(
        '--regression',
        choices=[True, False],
        type=lambda x: bool(strtobool(x)),
        default=True)
    parser.add_argument(
        '--onehot',
        choices=[True, False],
        type=lambda x: bool(strtobool(x)),
        default=False)
    return parser


def main():

    # CLI
    parser = cli(description='cli for pytorch fundamentals')
    args = parser.parse_args()

    # Instantiate data
    training_data = RandomNormalDataset(**vars(args))
    train_dataloader = DataLoader(
        training_data, batch_size=32, shuffle=True)

    test_data = RandomNormalDataset()
    test_dataloader = DataLoader(
        test_data, batch_size=32, shuffle=True)

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    main()
