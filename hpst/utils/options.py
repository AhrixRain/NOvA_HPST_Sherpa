import json
from argparse import Namespace

from typing import List, Optional


class Options(Namespace):
    def __init__(
            self,
            training_file: str = "",
            testing_file: str = "",
            validation_file: str = ""
    ):
        super(Options, self).__init__()

        # =========================================================================================
        # Network Architecture
        # =========================================================================================

        self.pointnet_patch_embed_depth = 1
        self.pointnet_patch_embed_channels = 512
        self.pointnet_num_heads = 16
        self.pointnet_patch_embed_groups = 6
        self.pointnet_patch_embed_neighbours = 8
        self.pointnet_enc_depths = [2,6,2]
        self.pointnet_enc_channels = [192, 384, 512]
        self.pointnet_enc_groups = [24, 48, 64]
        self.pointnet_enc_neighbours = [16, 16, 16]
        self.pointnet_dec_depths = [1, 1, 1]
        self.pointnet_dec_channels = [96, 192, 384]
        self.pointnet_dec_groups = [12, 24, 48]
        self.pointnet_dec_neighbours = [16, 16, 16]
        self.pointnet_grid_sizes = [16, 32, 64]
        self.dropout = 0.2

        # =========================================================================================
        # Dataset Options
        # =========================================================================================

        self.training_file: str = training_file
        self.testing_file: str = testing_file
        self.validation_file: str = validation_file

        # Limit the dataset to the first images% of the data.
        self.dataset_limit: float = 1.0

        # Percent of data to use for training vs. validation.
        self.train_validation_split: float = 0.95

        # Optional hard caps on the number of samples loaded for training/validation.
        self.train_max_samples: Optional[int] = None
        self.val_max_samples: Optional[int] = None

        # Training batch size.
        self.batch_size: int = 2048

        # Number of processes to spawn for data collection.
        self.num_dataloader_workers: int = 8

        self.normalize_features: bool = True

        # =========================================================================================
        # Training Options
        # =========================================================================================

        # The optimizer to use for trianing the network.
        # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix.
        self.optimizer: str = "AdamW"

        # Optimizer learning rate.
        self.learning_rate: float = 0.0001

        # Optimizer l2 penalty based on weight values.
        self.l2_penalty: float = 0.015

        # Clip the L2 norm of the gradient. Set to 0.0 to disable.
        self.gradient_clip: float = 90.0

        # Dropout added to all layers.
        self.dropout: float = 0.0

        # Number of epochs to train for.
        self.epochs: int = 25

        # Number of epochs to ramp up the learning rate up to the given value. Can be fractional.
        self.learning_rate_warmup_epochs: float = 1.0

        # Number of times to cycles the learning rate through cosine annealing with hard resets.
        # Set to 0 to disable cosine annealing and just use a decaying learning rate.
        self.learning_rate_cycles: int = 1

        # Total number of GPUs to use.
        self.num_gpu: int = 1

        self.event_prong_loss_proportion: float = 0.5

        # Current Used as the additional weight provided to the neutral target!!!
        self.loss_beta: float = 2.5

        # Exponent in the focal loss term (1 - p)
        self.loss_gamma: float = 0.0

        # Standard deviation of the noise to add to pixel-maps
        self.pixel_noise_std = 0.01

        # =========================================================================================
        # Miscellaneous Options
        # =========================================================================================

        # Whether or not to print additional information during training and log extra metrics.
        self.verbose_output: bool = True

        # Misc parameters used by sherpa to delegate GPUs and output directories.
        # These should not be set manually.
        self.usable_gpus: str = ''

        self.trial_time: str = ''

        self.trial_output_dir: str = './test_output'

    def update_options(self, new_options):
        integer_options = {key for key, val in self.__dict__.items() if isinstance(val, int)}
        boolean_options = {key for key, val in self.__dict__.items() if isinstance(val, bool)}
        for key, value in new_options.items():
            if key in integer_options:
                setattr(self, key, int(value))
            elif key in boolean_options:
                setattr(self, key, bool(value))
            else:
                setattr(self, key, value)

    @classmethod
    def load(cls, filepath: str):
        options = cls()
        with open(filepath, 'r') as json_file:
            options.update_options(json.load(json_file))
        return options

    def display(self):
        print("=" * 70)
        print("Options")
        print("-" * 70)
        for key, val in sorted(vars(self).items()):
            print(f"{key:32}: {val}")
        print("=" * 70)
