"""Script for PyTorch Model of Random Data.

* https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
* https://pytorch.org/tutorials/beginner/basics/intro.html
* 
* Cross Entropy for PyTorch expects label encoded vectors,
same as SparseCategoricalCross entropy in TensorFlow.
* Logits are more numerically stable and should be used in network training
Why?
"""

import argparse
from distutils.util import strtobool
import os
from typing import Callable, List
from numpy import dtype

from pytorch_lightning import LightningModule, Trainer

import torch
from torch import nn, Tensor
from torch.nn import Module
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
            onehot: bool = False,
            **kwargs):
        """Define state for RandomNormalDataset

        Args:
            samples: Number of samples
            x_dims: Dimensions for inputs.
            y_dims: Dimensions for labels.
            seed: Random seed.
            regression: True for real labels, else integer labels.
            onehot: True for one hotting labels, false otherwise.
        """

        # Inheritance
        super().__init__(**kwargs)

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
                low=0,
                high=y_dims,
                size=(samples, 1),
                requires_grad=False,
                dtype=torch.float32)
            if onehot:
                self.labels = F.one_hot(
                    self.labels, num_classes=y_dims).type(torch.float32)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx: int):
        input_at_idx = self.inputs[idx]
        label_at_idx = self.labels[idx]
        return input_at_idx, label_at_idx


class MLP(LightningModule):
    """Multilayer Perceptron.

    NOTE: Could use a regular `nn.Module` here, but the 
    `pl.LightningModule` is more well-organized and is akin to keras.
    """

    def __init__(
            self,
            x_dims: int,
            y_dims: int,
            loss_fn: Callable,
            regression: bool = True,
            onehot: bool = False,
            num_hidden_layers: int = 1,
            hidden_units: int = 32,
            **kwargs):
        """Define state for MLP."""

        # Required inheritance
        super().__init__(**kwargs)

        # Save args
        self.num_hidden_layers = num_hidden_layers
        self.loss_fn = loss_fn

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
            self.output_layer = nn.Linear(
                in_features=hidden_units, out_features=1)
        else:
            self.output_layer = nn.Linear(
                in_features=hidden_units, out_features=y_dims)

    def forward(self, x: Tensor) -> Tensor:
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

    def training_step(
            self, train_batch: List[Tensor], batch_idx: int) -> Tensor:
        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(
            self, val_batch: List[Tensor], batch_idx: int) -> None:
        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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
        '--max-epochs',
        type=int,
        default=2)
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
        'hidden_units', 'num_hidden_layers', 'batch_size', 'max_epochs']}

    # Args for model
    model_args = {k: v for k, v in vars(args).items() if k not in [
        'samples', 'batch_size', 'seed', 'max_epochs']}

    # Instantiate data
    # NOTE: Num workers must be spread out across data loading
    num_workers = os.cpu_count()
    training_data = RandomNormalDataset(**data_args)
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers//2)

    # Shuffle in validation data loader set to false per recommendations
    # of pytorch
    test_data = RandomNormalDataset(**data_args)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers//2)

    # # Inspect data loader data
    # batch_1_f, batch_1_t = next(iter(train_dataloader))
    # print(batch_1_f.size())
    # print(batch_1_t.size())
    # breakpoint()

    # Set loss function based on cli
    if not args.regression and args.y_dims == 2:
        loss_fn = nn.BCEWithLogitsLoss()
    elif not args.regression:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    # Instantiate network
    model = MLP(loss_fn=loss_fn, **model_args)

    # Instantiate trainer for abstracting model fitting
    gpus = torch.cuda.device_count()
    trainer = Trainer(max_epochs=args.max_epochs, gpus=gpus)

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,)


if __name__ == '__main__':
    main()
