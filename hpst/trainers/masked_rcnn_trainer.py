import numpy as np
import torch
from matplotlib import pyplot as plt
# from pytorch_lightning import metrics
import torchmetrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, jit

from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict

from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_scatter import scatter
from scipy.optimize import linear_sum_assignment

from hpst.utils.options import Options
from hpst.trainers.neutrino_base import NeutrinoBase

from hpst.dataset.bbox_mask_dataset import BboxMaskDataset
from hpst.models.mask_rcnn import MaskRCNN

from hpst.utils.learning_rate_schedules import get_linear_schedule_with_warmup
from hpst.utils.learning_rate_schedules import get_cosine_with_hard_restarts_schedule_with_warmup

import sys

TArray = np.ndarray

# used to be List[List[Tuple[Tensor, Tensor, Tensor]]]
# @torch.jit.script
def collate_sparse(data: List[Tuple[int, Tuple[Tensor,Dict[str,Tensor]], Tuple[Tensor,Dict[str, Tensor]]]]):
    hits_index = torch.tensor([d[0] for d in data])

    features_x = ([d[1][0] for d in data])
    targets_x = [d[1][1] for d in data]

    features_y = ([d[2][0] for d in data])
    targets_y = [d[2][1] for d in data]

    return hits_index, features_x, targets_x, features_y, targets_y


class MaskedRNNTrainer(NeutrinoBase):
    def __init__(self, options: Options, train_perc=1.0):
        """

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(MaskedRNNTrainer, self).__init__(options, train_perc=train_perc)

        self.automatic_optimization = False

        self.num_objects = 10
        self.num_classes = self.training_dataset.num_classes
        self.network_x = MaskRCNN(num_classes=self.num_classes)
        self.network_y = MaskRCNN(num_classes=self.num_classes)

        """
        self.beta = self.options.loss_beta
        effective_num = 1.0 - self.beta**self.training_dataset.target_count
        self.weights = (1.0 - self.beta) / effective_num
        self.weights = self.weights / self.weights.sum() * self.training_dataset.num_classes

        self.gamma = options.loss_gamma
        if self.options.loss_beta < 0.01:
            self.beta = 1 - 1 / len(self.training_dataset)
        """

    @property
    def dataset(self):
        return BboxMaskDataset
    
    @property
    def dataloader_options(self):
        return {
            "drop_last": False,
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers,
            "collate_fn": collate_sparse
        }

    def forward(self, features_x, features_y, targets_x=None, targets_y=None) -> Tensor:
        # Normalize the high level layers
        """
        features1 = features1.clone()
        features1 -= self.mean
        features1 /= self.std

        features2 = features2.clone()
        features2 -= self.mean
        features2 /= self.std
        """

        outputs1 = self.network_x(features_x, targets=targets_x)
        outputs2 = self.network_y(features_y, targets=targets_y)

        return outputs1, outputs2
    
    def bipartite_loss(self, logits, targets, batches):
        object_preds = -F.log_softmax(logits, dim=-1)
        batch_size = batches.max()
        batch_size = batch_size + 1
        cost_matrix = scatter(object_preds, (batches*self.num_objects)+targets, dim_size=self.num_objects*batch_size, reduce="sum", dim=0)
        cost_matrix = cost_matrix.reshape((batch_size, self.num_objects, -1))
        cpu_cm = cost_matrix.detach().cpu().numpy()
        row_inds, col_inds, indices = [], [], []
        for i, cm in enumerate(cpu_cm):
            row_ind, col_ind = linear_sum_assignment(cm, maximize=False)
            row_ind = torch.from_numpy(row_ind)
            col_ind = torch.from_numpy(col_ind)
            index = i*torch.ones_like(row_ind)
            row_inds.append(row_ind)
            col_inds.append(col_ind)
            indices.append(index)

        row_inds = torch.cat(row_inds, dim=0).to(cost_matrix.device)
        col_inds = torch.cat(col_inds, dim=0).to(cost_matrix.device)
        indices = torch.cat(indices, dim=0).to(cost_matrix.device)

        return cost_matrix[indices, row_inds, col_inds].sum()/object_preds.shape[0]
    
    def bipartite_accuracy(self, object_preds, targets, batches):
        object_preds = F.one_hot(object_preds, num_classes=self.num_objects)
        batch_size = batches.max()
        batch_size = batch_size + 1
        cost_matrix = scatter(object_preds, (batches*self.num_objects)+targets, dim_size=self.num_objects*batch_size, reduce="sum", dim=0)
        cost_matrix = cost_matrix.reshape((batch_size, self.num_objects, -1))
        cpu_cm = cost_matrix.detach().cpu().numpy()
        row_inds, col_inds, indices = [], [], []
        for i, cm in enumerate(cpu_cm):
            row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
            row_ind = torch.from_numpy(row_ind)
            col_ind = torch.from_numpy(col_ind)
            index = i*torch.ones_like(row_ind)
            row_inds.append(row_ind)
            col_inds.append(col_ind)
            indices.append(index)

        row_inds = torch.cat(row_inds, dim=0).to(cost_matrix.device)
        col_inds = torch.cat(col_inds, dim=0).to(cost_matrix.device)
        indices = torch.cat(indices, dim=0).to(cost_matrix.device)

        return cost_matrix[indices, row_inds, col_inds].sum()/object_preds.shape[0]

    def training_step(self, batch, batch_idx):
        opt_x, opt_y = self.optimizers()
        (_, features_x, targets_x, features_y, targets_y) = batch

        #print(features_x.shape, targets_x[0]["boxes"].shape, targets_x[0]["labels"].shape, targets_x[0]["masks"].shape)

        output1, output2 = self.forward(features_x, features_y, targets_x, targets_y)

        loss1 = output1["loss_classifier"] + output1["loss_box_reg"] + output1["loss_mask"] + output1["loss_objectness"] + output1["loss_rpn_box_reg"]

        opt_x.zero_grad()
        self.manual_backward(loss1)
        opt_x.step()

        loss2 = output2["loss_classifier"] + output2["loss_box_reg"] + output2["loss_mask"] + output2["loss_objectness"] + output2["loss_rpn_box_reg"]

        opt_y.zero_grad()
        self.manual_backward(loss2)
        opt_y.step()

        if self.trainer.is_last_batch:
            sch_x, sch_y = self.lr_schedulers()
            sch_x.step()
            sch_y.step()

        #return (loss1 + loss2).item()

    def validation_step(self, batch, batch_idx):
        (_, features_x, targets_x, features_y, targets_y) = batch

        output1, output2 = self.forward(features_x, features_y)

        correct = 0
        count = 0
        for o, t in zip(output1, targets_x):
            if o["masks"].shape[0] > 0:
                true_labels = torch.argmax(t["masks"], axis=0)
                hits = torch.argmax(o["masks"], axis=0)[0, true_labels != 0]
                predictions = o["labels"][hits]
                true_labels = true_labels[true_labels != 0]
                correct += (predictions != true_labels).sum()
                count += predictions.shape[0]

        for o, t in zip(output2, targets_y):
            if o["masks"].shape[0] > 0:
                true_labels = torch.argmax(t["masks"], axis=0)
                hits = torch.argmax(o["masks"], axis=0)[0, true_labels != 0]
                predictions = o["labels"][hits]
                true_labels = true_labels[true_labels != 0]
                correct += (predictions != true_labels).sum()
                count += predictions.shape[0]

        if count > 0:
            self.log("val_accuracy", correct/count, sync_dist=True, batch_size=self.options.batch_size)
        else:
            self.log("val_accuracy", 0, sync_dist=True, batch_size=self.options.batch_size)
    
    def configure_optimizers(self):
        optimizer = None

        
        optimizer = getattr(torch.optim, self.options.optimizer)

        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        optimizer1 = optimizer(self.network_x.parameters(), lr=self.options.learning_rate, weight_decay=self.options.l2_penalty)

        if self.options.learning_rate_cycles < 1:
            scheduler1 = get_linear_schedule_with_warmup(
                 optimizer1,
                 num_warmup_steps=self.warmup_steps,
                 num_training_steps=self.total_steps
             )
        else:
            scheduler1 = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer1,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler1 = {
            'scheduler': scheduler1,
            'interval': 'step',
            'frequency': 1
        }

        optimizer2 = optimizer(self.parameters(), lr=self.options.learning_rate, weight_decay=self.options.l2_penalty)

        if self.options.learning_rate_cycles < 1:
            scheduler2 = get_linear_schedule_with_warmup(
                 optimizer2,
                 num_warmup_steps=self.warmup_steps,
                 num_training_steps=self.total_steps
             )
        else:
            scheduler2 = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer2,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler2 = {
            'scheduler': scheduler2,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer1, optimizer2], [scheduler1, scheduler2]