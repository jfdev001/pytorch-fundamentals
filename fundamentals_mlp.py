"""Script for PyTorch/PyTorchLightning model of random data.

NOTE: Could remove the `**data_args` logic by just making the
data an field of the model itself.

Main Docs/Tutorials:
* https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
* https://pytorch.org/tutorials/beginner/basics/intro.html

Other Docs:
* hparam and argparse: https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html

Notes:
* Cross Entropy for PyTorch expects label encoded vectors,
same as SparseCategoricalCross entropy in TensorFlow.
* Logits are more numerically stable and should be used in network training
"""

import argparse
from argparse import ArgumentParser
from distutils.util import strtobool
import os
from typing import Callable, List

from pytorch_lightning import LightningModule, Trainer

import torch
from torch import nn, Tensor
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
            **kwargs):
        """Define state for RandomNormalDataset.

        Args:
            samples: Number of samples
            x_dims: Dimensions for inputs.
            y_dims: Dimensions for labels.
            seed: Random seed.
            regression: True for real labels, else integer labels.
        """

        # Inheritance
        super().__init__(**kwargs)

        # Save args
        self.samples = samples

        # Set seed
        torch.manual_seed(seed)

        # Define real inputs
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
            # if onehot:
            #     self.labels = F.one_hot(
            #         self.labels, num_classes=y_dims).type(torch.float32)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx: int):
        input_at_idx = self.inputs[idx]
        label_at_idx = self.labels[idx]
        return input_at_idx, label_at_idx


class MLP(LightningModule):
    """Multilayer Perceptron.

    Encapsulates both training steps and random data creation.
    """

    def __init__(
            self,
            x_dims: int,
            y_dims: int,
            regression: bool = True,
            samples: int = 128,
            batch_size: int = 32,
            max_epochs: int = 2,
            seed: int = 0,
            num_hidden_layers: int = 1,
            hidden_units: int = 32,
            learning_rate: float = 1e-3,
            save_hparams: bool = True,
            **kwargs):
        """Define state for MLP."""

        # Required inheritance
        super().__init__(**kwargs)

        # Set loss function based on cur hyperparameters
        if not regression and y_dims == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif not regression:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # Save args
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.regression = regression
        self.num_hidden_layers = num_hidden_layers
        self.samples = samples
        self.seed = seed
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        # Set num workers attr
        self.num_workers = os.cpu_count()

        # Set data
        self.train_data_loader = None
        self.test_data_loader = None
        self._init_dataloaders()

        # Save hparams or not
        # NOTE: Must be serializable args... so Callables and custom objs
        # probably aren't good
        if save_hparams:
            self.save_hyperparameters()

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
        """Forward pass and loss on a batch of training data."""

        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(
            self, val_batch: List[Tensor], batch_idx: int) -> None:
        """Forward pass and loss on a batch of validation data."""

        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return None

    def configure_optimizers(self) -> Callable:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _init_dataloaders(self) -> None:
        """Creates dataloaders for use with `pl.Trainer` later."""

        # Instantiate training data
        training_data = RandomNormalDataset(
            samples=self.samples,
            x_dims=self.x_dims,
            y_dims=self.y_dims,
            seed=self.seed,
            regression=self.regression)

        self._train_dataloader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers//2)

        # Shuffle in validation data loader set to false per recommendations
        # of pytorch... reproducibility presumably since order of
        # testing here doesn't matter... model set to inference mode (i.e.,
        # no parameter updates)
        # so independent, and identical distributed (iid) assumption
        # not needed for grad descent
        test_data = RandomNormalDataset(
            samples=self.samples,
            x_dims=self.x_dims,
            y_dims=self.y_dims,
            seed=self.seed,
            regression=self.regression)

        self._test_dataloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers//2)

    @staticmethod
    def add_model_specific_args(
            parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to CLI object."""

        parser = parent_parser.add_argument_group(
            'MLP',
            'params for multilayer perceptron model')
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
            '--hidden-units',
            type=int,
            default=32)
        parser.add_argument(
            '--num-hidden-layers',
            type=int,
            default=1)
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-3)
        parser.add_argument(
            '--save-hparams',
            choices=[True, False],
            type=lambda x: bool(strtobool(x)),
            default=True)

        return parent_parser


def cli(description: str):
    """Command line interface for fundamentals of pytorch.

    https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    """

    parser = argparse.ArgumentParser(description=description)

    # # Add trainer args to the parser
    # parser = Trainer.add_argparse_args(parser)

    # Add model args to parser
    parser = MLP.add_model_specific_args(parser)

    return parser


def main():

    # CLI
    parser = cli(description='cli for pytorch fundamentals')
    args = parser.parse_args()

    # # Get keys of trainer so that those can be ignored...
    # action_groups = parser._action_groups
    # num_groups = len(action_groups)
    # trainer_args_not_found = True
    # group_ix = 0
    # bad_keys = []
    # while trainer_args_not_found and group_ix < num_groups:
    #     group = action_groups[group_ix]
    #     if group.title == 'pl.Trainer':
    #         trainer_args_not_found = False
    #         group_dict = {a.dest: getattr(args, a.dest, None)
    #                       for a in group._group_actions}
    #         bad_keys = list(group_dict.keys())
    #     group_ix += 1

    # Instantiate network
    model = MLP(**vars(args))

    # Instantiate trainer for abstracting model fitting
    gpus = torch.cuda.device_count()
    trainer = Trainer(max_epochs=model.max_epochs, gpus=gpus, callbacks=None)

    # Train model
    trainer.fit(
        model,
        train_dataloaders=model._train_dataloader,
        val_dataloaders=model._test_dataloader,)


if __name__ == '__main__':
    main()
