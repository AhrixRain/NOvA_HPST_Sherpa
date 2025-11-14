"""
GAT based model
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
# from pytorch_lightning import metrics
import torchmetrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, jit

from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union

from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_scatter import scatter
from scipy.optimize import linear_sum_assignment

from hpst.utils.options import Options
from hpst.trainers.neutrino_base import NeutrinoBase

from hpst.dataset.heterogenous_sparse_dataset import HeterogenousSparseDataset
from hpst.models.gat import GATInterface

import sys

TArray = np.ndarray

# used to be List[List[Tuple[Tensor, Tensor, Tensor]]]
@torch.jit.script
def collate_sparse(data: List[Tuple[int, List[Tensor], List[Tensor]]]):
    hits_index = torch.tensor([d[0] for d in data])
    features1 = torch.cat([d[1][0] for d in data])
    coordinates1 = torch.cat([d[1][1] for d in data])
    targets1 = torch.cat([d[1][2] for d in data])
    object_targets1 = torch.cat([d[1][3] for d in data])
    features2 = torch.cat([d[2][0] for d in data])
    coordinates2 = torch.cat([d[2][1] for d in data])
    targets2 = torch.cat([d[2][2] for d in data])
    object_targets2 = torch.cat([d[2][3] for d in data])
    batches1 = torch.cat([i*torch.ones((d[1][0].shape[0],), device=d[1][0].device) for i,d in enumerate(data)]).to(torch.int64)
    batches2 = torch.cat([i*torch.ones((d[2][0].shape[0],), device=d[2][0].device) for i,d in enumerate(data)]).to(torch.int64)
    return (hits_index, batches1, features1, coordinates1, targets1, object_targets1, batches2, features2, coordinates2, targets2, object_targets2)


class GATTrainer(NeutrinoBase):
    def __init__(self, options: Options, train_perc=None, num_objects=10):
        """

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(GATTrainer, self).__init__(options, train_perc=train_perc)

        self.num_objects = num_objects
        self.num_classes = self.training_dataset.num_classes
        self.network = GATInterface(options,
                            self.training_dataset.num_features,
                            self.num_classes + self.num_objects)

        self.beta = self.options.loss_beta
        effective_num = 1.0 - self.beta**self.training_dataset.target_count
        self.weights = (1.0 - self.beta) / effective_num
        self.weights = self.weights / self.weights.sum() * self.training_dataset.num_classes

        self.gamma = options.loss_gamma
        if self.options.loss_beta < 0.01:
            self.beta = 1 - 1 / len(self.training_dataset)

    @property
    def dataset(self):
        return HeterogenousSparseDataset
    
    @property
    def dataloader_options(self):
        return {
            "drop_last": True,
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers,
            "collate_fn": collate_sparse
        }

    def forward(self, features1: Tensor, coords1: Tensor, batch1: Tensor, features2: Tensor, coords2: Tensor, batch2: Tensor) -> Tensor:
        # Normalize the high level layers
        features1 = features1.clone()
        features1 -= self.mean
        features1 /= self.std

        features2 = features2.clone()
        features2 -= self.mean
        features2 /= self.std

        outputs1, outputs2 = self.network((coords1, features1, batch1), (coords2, features2, batch2))
        
        predictions1, object_predictions1, predictions2, object_predictions2 = outputs1[:, :self.num_classes], outputs1[:, self.num_classes:], outputs2[:, :self.num_classes], outputs2[:, self.num_classes:]

        return predictions1, object_predictions1, predictions2, object_predictions2
    
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
        (_, batches1, features1, coordinates1, targets1, object_targets1, batches2, features2, coordinates2, targets2, object_targets2) = batch

        predictions1, object_predictions1, predictions2, object_predictions2 = self.forward(features1, coordinates1, batches1, features2, coordinates2, batches2)
        
        mask1 = ((targets1 != -1))
        mask2 = ((targets2 != -1))

        # mask_group = (scatter((~mask).to(int), batch, dim=-1, reduce='sum') == 0)
        # mask = torch.gather(mask_group, 0, batch)

        predictions1 = predictions1[mask1]
        object_predictions1 = object_predictions1[mask1]
        targets1 = targets1[mask1]
        object_targets1 = object_targets1[mask1]
        batches1 = batches1[mask1]
        predictions2 = predictions2[mask2]
        object_predictions2 = object_predictions2[mask2]
        targets2 = targets2[mask2]
        batches2 = batches2[mask2]
        object_targets2 = object_targets2[mask2]

        ce_loss1 = F.cross_entropy(predictions1, targets1, weight=self.weights.to(predictions1.device))
        ce_loss2 = F.cross_entropy(predictions2, targets2, weight=self.weights.to(predictions2.device))

        mask1 = ((object_targets1 != -1) & (object_targets1 < self.num_objects))
        mask2 = ((object_targets2 != -1) & (object_targets2 < self.num_objects))

        predictions1 = predictions1[mask1]
        object_predictions1 = object_predictions1[mask1]
        targets1 = targets1[mask1]
        object_targets1 = object_targets1[mask1]
        batches1 = batches1[mask1]
        predictions2 = predictions2[mask2]
        object_predictions2 = object_predictions2[mask2]
        targets2 = targets2[mask2]
        batches2 = batches2[mask2]
        object_targets2 = object_targets2[mask2]

        object_loss = self.bipartite_loss(
            torch.cat((object_predictions1, object_predictions2), dim=0),
            torch.cat((object_targets1, object_targets2), dim=0),
            torch.cat((batches1, batches2), dim=0)
        )

        # object_loss is averaged over both sets while cross entropy is averaged over each one
        # so it should be roughly half of the size of the cross entropy losses
        loss = 2*object_loss + ce_loss1 + ce_loss2

        self.log("object_loss", object_loss, batch_size=self.options.batch_size)
        self.log("train_loss", ce_loss1 + ce_loss2, batch_size=self.options.batch_size)
        self.log("total_train_loss", loss, batch_size=self.options.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        (_, batches1, features1, coordinates1, targets1, object_targets1, batches2, features2, coordinates2, targets2, object_targets2) = batch

        predictions1, object_predictions1, predictions2, object_predictions2 = self.forward(features1, coordinates1, batches1, features2, coordinates2, batches2)

        # If there's more than self.num_objects objects in the scene, then we ignore the rest
        mask1 = ((targets1 != -1) & (object_targets1 != -1) & (object_targets1 < self.num_objects))
        mask2 = ((targets2 != -1) & (object_targets2 != -1) & (object_targets2 < self.num_objects))

        predictions1 = predictions1[mask1]
        object_predictions1 = object_predictions1[mask1]
        targets1 = targets1[mask1]
        object_targets1 = object_targets1[mask1]
        batches1 = batches1[mask1]
        predictions2 = predictions2[mask2]
        object_predictions2 = object_predictions2[mask2]
        targets2 = targets2[mask2]
        object_targets2 = object_targets2[mask2]
        batches2 = batches2[mask2]
        
        _, predictions1 = torch.max(predictions1, dim=-1)
        _, predictions2 = torch.max(predictions2, dim=-1)
        _, object_predictions1 = torch.max(object_predictions1, dim=-1)
        _, object_predictions2 = torch.max(object_predictions2, dim=-1)

        accuracy = ((predictions1 == targets1).to(float).sum() + (predictions2 == targets2).to(float).sum())/(len(predictions1) + len(predictions2))
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.options.batch_size)
        object_accuracy = self.bipartite_accuracy(
            torch.cat((object_predictions1, object_predictions2), dim=0),
            torch.cat((object_targets1, object_targets2), dim=0),
            torch.cat((batches1, batches2), dim=0)
        )
        # ((object_predictions1 == object_targets1).to(float).sum() + (object_predictions2 == object_targets2).to(float).sum())/(len(object_predictions1) + len(object_predictions2))
        self.log("object_accuracy", object_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.options.batch_size)

        return metrics