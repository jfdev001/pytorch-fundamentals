"""Script for PyTorch Model of Random Data.

NOTE: Cross Entropy for PyTorch expects label encoded vectors,
same as SparseCategoricalCross entropy in TensorFlow.
"""

import argparse
from distutils.util import strtobool

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
        torch.manual_seed(seed)

        # Define real inputs
        # NOTE: Gradient requirements??? I think no
        # because grad descent is dC/dW or dC/dBias
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

    def __getitem__(self, idx: int):
        input_at_idx = self.inputs[idx]
        label_at_idx = self.labels[idx]
        return input_at_idx, label_at_idx


class MLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(
            self,
            x_dims: int,
            y_dims: int,
            regression: bool = True,
            onehot: bool = False,
            num_hidden_layers: int = 1,
            hidden_units: int = 32,):
        """Define state for MLP."""

        # Save args
        self.num_hidden_layers = num_hidden_layers

        # Required inheritance
        super().__init__()

        # Initial hidden lyaer
        self.init_hidden = nn.Sequential(
            nn.Linear(in_features=x_dims, out_features=hidden_units),
            nn.ReLU())

        # Deeper hidden net
        if num_hidden_layers > 1:
            linear_relu_modules = []

            for lyr in range(num_hidden_layers - 1):
                linear_relu_modules.append(nn.Linear(
                    in_features=hidden_units, out_features=hidden_units))
                linear_relu_modules.append(nn.ReLU())

            self.linear_relu_stack = nn.Sequential(*linear_relu_modules)

        # Output layer
        if not regression and y_dims == 2:
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=hidden_units, out_features=1),
                nn.Sigmoid())
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=hidden_units, out_features=y_dims),)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Inputs to neural network of shape (m_samples, n_dims)
        Returns:
            Predictions.
        """

        # Initial projection
        hidden_proj = self.init_hidden(x)

        # Deeper projection
        if self.num_hidden_layers > 1:
            hidden_proj = self.linear_relu_stack(hidden_proj)

        # Output projection
        preds = self.output_layer(hidden_proj)

        # Result of forward computation
        return preds


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
        '--batch-size',
        type=int,
        default=32)
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
    parser.add_argument(
        '--hidden-units',
        type=int,
        default=32)
    parser.add_argument(
        '--num-hidden-layers',
        type=int,
        default=1)
    return parser


def main():

    # CLI
    parser = cli(description='cli for pytorch fundamentals')
    args = parser.parse_args()

    # Args for data
    data_args = {k: v for k, v in vars(args).items() if k not in [
        'hidden_units', 'num_hidden_layers', 'batch_size']}

    # Args for model
    model_args = {k: v for k, v in vars(args).items() if k not in [
        'samples', 'batch_size', 'seed']}

    # Instantiate data
    training_data = RandomNormalDataset(**data_args)
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True)

    test_data = RandomNormalDataset()
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True)

    # Define gpu/cpu device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate network
    model = MLP(**model_args).to(device=device)


if __name__ == '__main__':
    main()
