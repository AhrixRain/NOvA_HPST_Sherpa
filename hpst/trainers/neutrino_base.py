import lightning.pytorch as pl
import torch

# noinspection PyProtectedMember
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from hpst.utils.options import Options
from hpst.utils.learning_rate_schedules import get_linear_schedule_with_warmup
from hpst.utils.learning_rate_schedules import get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_annealing_with_warmup
import torch_geometric.transforms as T


class _DatasetSubset(Dataset):
    """Lightweight proxy that restricts a dataset to a predefined list of indices."""

    def __init__(self, dataset: Dataset, indices: torch.Tensor):
        super().__init__()
        self._dataset = dataset
        self._indices = indices.to(torch.long)

    def __len__(self) -> int:
        return self._indices.numel()

    def __getitem__(self, idx):
        return self._dataset[int(self._indices[idx])]

    def __getattr__(self, item):
        return getattr(self._dataset, item)


def _limit_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if (
        dataset is None
        or max_samples is None
        or max_samples <= 0
        or len(dataset) <= max_samples
    ):
        return dataset
    indices = torch.randperm(len(dataset))[:max_samples]
    print(f"Limiting dataset to {len(indices)} samples (was {len(dataset)}).")
    return _DatasetSubset(dataset, indices)


class NeutrinoBase(pl.LightningModule):
    def __init__(self, options: Options, train_perc=None):
        super(NeutrinoBase, self).__init__()

        # self.hparams = options
        self.options = options
        self.train_perc = train_perc
        
        self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()

        self.mean = 0
        self.std = 1

        self.extra_mean = 0
        self.extra_std = 1

        self.pixel_mean = 0
        self.pixel_std = 1

        # Normalize datasets using training dataset statistics
        if self.options.normalize_features:
            (self.mean, self.std,
             self.extra_mean, self.extra_std,
             self.pixel_mean, self.pixel_std) = self.training_dataset.compute_statistics()

            self.mean = torch.nn.Parameter(self.mean, requires_grad=False)
            self.std = torch.nn.Parameter(self.std, requires_grad=False)

            self.extra_mean = torch.nn.Parameter(self.extra_mean, requires_grad=False)
            self.extra_std = torch.nn.Parameter(self.extra_std, requires_grad=False)

            if self.training_dataset.pixels is not None:
                self.pixel_mean = torch.nn.Parameter(self.pixel_mean, requires_grad=False)
                self.pixel_std = torch.nn.Parameter(self.pixel_std, requires_grad=False)

        train_cap = getattr(self.options, "train_max_samples", None)
        val_cap = getattr(self.options, "val_max_samples", None)
        self.training_dataset = _limit_dataset(self.training_dataset, train_cap)
        self.validation_dataset = _limit_dataset(self.validation_dataset, val_cap)

        self.steps_per_epoch = len(self.training_dataset) // (self.options.batch_size * max(1, self.options.num_gpu))
        self.total_steps = self.steps_per_epoch * self.options.epochs
        self.warmup_steps = int(round(self.steps_per_epoch * self.options.learning_rate_warmup_epochs))

    @property
    def dataset(self):
        return Dataset

    @property
    def dataloader(self):
        return DataLoader

    @property
    def dataloader_options(self):
        return {
            "drop_last": True,
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers
        }

    def configure_optimizers(self):
        optimizer = None

        
        optimizer = getattr(torch.optim, self.options.optimizer)

        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        optimizer = optimizer(self.parameters(), lr=self.options.learning_rate, weight_decay=self.options.l2_penalty)

        if self.options.learning_rate_cycles < 1:
            scheduler = get_linear_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=self.warmup_steps,
                 num_training_steps=self.total_steps
             )
        else:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def create_datasets(self):
        transform = T.Compose([
            # T.RandomJitter(5),
            T.RandomScale((0.99,1.01)),
            T.RandomRotate((-1,1),axis=0),
            T.RandomRotate((-1,1),axis=1),
            # T.RandomShear(0.1),
        ])
        print("loading training")
        if len(self.options.validation_file) > 0:
            training_dataset = self.dataset(self.options.training_file, transform=transform)
            validation_dataset = self.dataset(self.options.validation_file)
        else:
            # Compute the training / validation ranges based on the data-split and the limiting percentage.
            train_validation_split = self.options.dataset_limit * self.options.train_validation_split
            if self.train_perc is None:
                train_range = (0.0, train_validation_split)
                validation_range = (train_validation_split, self.options.dataset_limit)
            else:
                train_range = (0.0, train_validation_split*self.train_perc)
                validation_range = (train_validation_split, train_validation_split + self.train_perc*(self.options.dataset_limit - train_validation_split))

            training_dataset = self.dataset(self.options.training_file, train_range, transform=transform)
            validation_dataset = self.dataset(self.options.training_file, validation_range)
        
        print("loading testing")
        testing_dataset = None
        if len(self.options.testing_file) > 0:
            test_range = (0.0, 1.0)
            testing_dataset = self.dataset(self.options.testing_file, test_range, testing=True)

        return training_dataset, validation_dataset, testing_dataset

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.training_dataset, shuffle=True, **self.dataloader_options)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.validation_dataset, **self.dataloader_options)

    def test_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise ValueError("Testing dataset not provided.")

        return self.dataloader(self.testing_dataset, **self.dataloader_options)
