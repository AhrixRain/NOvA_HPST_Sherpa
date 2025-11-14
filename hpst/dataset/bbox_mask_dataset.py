from typing import Tuple, List, Optional

import h5py
import numba
import numpy as np
import torch
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_scatter import segment_csr

from torch.utils.data import Dataset
from tqdm import tqdm
from torch_scatter import scatter

# taken from Point Transformer V2
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

class BboxMaskDataset(Dataset):
    def __init__(self, data_file: str, limit_index=1.0,  transform=None, testing=False):
        super(BboxMaskDataset, self).__init__()

        self.pixel_mean = 0
        self.pixel_std = 1
        self.pixels = None

        # Needed to make compatible with other datasets
        self.mean = 0
        self.std = 1

        self.extra_mean = 0
        self.extra_std = 1

        #test_idxs = torch.load("/home/roblesee/dune/UCI-NuML-Codebase/NOvA/Transformer/test_idxs.pt")

        with h5py.File(data_file, 'r') as file:
            # the masking operation in h5py is abysmally slow
            # under no circumstances try to do numpy operations
            # directly to the h5py file, instead load the numpy
            # array into memory and do that instead
            self.num_events = file["cvnmap_index"][:,0].max()
            # hitlim_mask = np.nonzero(nhit)
            # masked_hits = nhit[hitlim_mask]
            # cumsum_hits = np.cumsum(masked_hits)
            # self.hitlim = np.pad(cumsum_hits, (1,0), 'constant')
    
            hits_index = file["cvnmap_index"][:,0]
            hit_differences = torch.argwhere(torch.from_numpy(hits_index[1:] != hits_index[:-1]))[:,0]+1
            self.hitlim = np.pad(hit_differences, (1,0), 'constant')
            limit_index = self.compute_limit_index(limit_index)
            self.min_limit = limit_index.min()
            self.max_limit = limit_index.max()

            self.id_hits_index = torch.from_numpy(file["cvnmap_index"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit], 0].astype(np.int64))
            self.hits_index = torch.from_numpy(file["cvnmap_index"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit], 0].astype(np.int64))
            self.features = torch.from_numpy(file["cvnmap_value"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit]].astype(np.float32).reshape((-1,1)))
            self.targets = torch.from_numpy(file["cvnmap_label"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit]].astype(np.int64)).to(int)
            self.object_targets = torch.from_numpy(file["cvnmap_object"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit]].astype(np.int64)).to(int) # + 1
            coords = torch.from_numpy(file["cvnmap_index"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit], 1].astype(np.float32))
            img_width, img_height = 100, 80
            self.hitview = coords // (img_width * img_height)
            X = (coords % (img_width*img_height)) // img_height
            YZ = coords % img_height
            self.coordinates = torch.stack([X, YZ], dim=-1).to(torch.int64)
            #test_idxs -= self.hitlim[self.min_limit]
            self.hitlim = torch.from_numpy((self.hitlim[self.min_limit:self.max_limit+1] - self.hitlim[self.min_limit]).astype(int))
            hitlim_lo = self.hitlim[:-1]
            hitlim_hi = self.hitlim[1:]
            self.hits_index = self.hits_index - self.hits_index[0]
        
        self.features = (self.features - self.features.amin()) / (self.features.amax() - self.features.amin())

        # we need to reindex the samples starting from 0, so we use the hitlim:
        num_segments = torch.zeros((self.hits_index.amax()+1,), dtype=self.object_targets.dtype) \
            .scatter_reduce_(dim=0, index=self.hits_index, src=self.object_targets, reduce="amax") 
        self.bbox_offsets = torch.zeros((num_segments.shape[0]+1,), dtype=num_segments.dtype)
        torch.cumsum(num_segments+1, out=self.bbox_offsets[1:], dim=0)
        #self.bbox_offsets = self.bbox_offsets[:-1]

        object_target_index = self.bbox_offsets[self.hits_index] + self.object_targets
        object_target_index_x = object_target_index[self.hitview==0]

        #index_offsets_x = self.bbox_offsets[self.hits_index[self.hitview==0]]
        #object_target_index_x = index_offsets_x + self.object_targets[self.hitview==0]
        bbox_min_x1 = torch.zeros((object_target_index_x.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_x, src=self.coordinates[self.hitview==0,0], reduce="amin", include_self=False)
        bbox_min_x2 = torch.zeros((object_target_index_x.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_x, src=self.coordinates[self.hitview==0,1], reduce="amin", include_self=False)
        bbox_max_x1 = torch.zeros((object_target_index_x.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_x, src=self.coordinates[self.hitview==0,0], reduce="amax", include_self=False)
        bbox_max_x2 = torch.zeros((object_target_index_x.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_x, src=self.coordinates[self.hitview==0,1], reduce="amax", include_self=False)
        self.bbox_x = torch.stack([bbox_min_x2-1, bbox_min_x1-1, bbox_max_x2+1, bbox_max_x1+1], dim=-1).to(torch.float32) # (64*N, 2), (64*N, 2) -> (64*N, 4)
        self.bbox_x[self.bbox_x[:,0] < 0,0] = 0
        self.bbox_x[self.bbox_x[:,1] < 0,1] = 0
        self.bbox_x[self.bbox_x[:,2] > img_height-1,2] = img_height-1
        self.bbox_x[self.bbox_x[:,3] > img_width-1,3] = img_width-1
        
        #index_offsets_y = self.bbox_offsets[self.hits_index[self.hitview==1]]
        #object_target_index_y = index_offsets_y + self.object_targets[self.hitview==1]
        object_target_index_y = object_target_index[self.hitview==1]
        bbox_min_y1 = torch.zeros((object_target_index_y.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_y, src=self.coordinates[self.hitview==1,0], reduce="amin", include_self=False)
        bbox_min_y2 = torch.zeros((object_target_index_y.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_y, src=self.coordinates[self.hitview==1,1], reduce="amin", include_self=False)
        bbox_max_y1 = torch.zeros((object_target_index_y.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_y, src=self.coordinates[self.hitview==1,0], reduce="amax", include_self=False)
        bbox_max_y2 = torch.zeros((object_target_index_y.amax()+1,), dtype=self.coordinates.dtype) \
            .scatter_reduce_(dim=0, index=object_target_index_y, src=self.coordinates[self.hitview==1,1], reduce="amax", include_self=False)
        self.bbox_y = torch.stack([bbox_min_y2-1, bbox_min_y1-1, bbox_max_y2+1, bbox_max_y1+1], dim=-1).to(torch.float32) # (64*N, 2), (64*N, 2) -> (64*N, 4)
        self.bbox_y[self.bbox_y[:,0] < 0,0] = 0
        self.bbox_y[self.bbox_y[:,1] < 0,1] = 0
        self.bbox_y[self.bbox_y[:,2] > img_height-1,2] = img_height-1
        self.bbox_y[self.bbox_y[:,3] > img_width-1,3] = img_width-1
    
        self.num_features = self.features.shape[-1]
        # We're ignoring cosmic events for now
        self.num_classes = 10 # self.targets.max().item() + 1
        self.transform = transform
        self.target_count = torch.bincount(self.targets[(self.targets != -1)].to(int))

        merge_labels = True
        if merge_labels:
            names = {
                0: "background",
                1: "muon",
                2: "electron",
                3: "proton",
                4: "photon",
                5: "pion"
            }
            mapping = {
                0: 0, # empty -> background
                1: 2,  # electron -> electron
                2: 1, # muon -> muon
                3: 3, # proton -> proton
                4: 0, # neutron -> ignore
                5: 5, # Pion -> Pion
                6: 0, # PiZero/Neutral Pion -> ignore
                7: 4, # Gamma/Photon -> Photon
                8: 3, # MostlyHA -> proton
                9: 2, # MostlyEM -> electron
                10: 1, # MostlyMU -> muon
                11: 0 # Unknown -> ignore
            }
            targets_copy = self.targets.clone()

            for key, value in mapping.items():
                self.targets[targets_copy == key] = value
            self.target_count = torch.bincount(self.targets[self.targets != -1].to(int))
            self.num_classes = 6

        self.targets_x = torch.zeros((object_target_index_x.amax()+1,), dtype=self.object_targets.dtype) \
            .scatter_(dim=0, index=object_target_index_x, src=self.targets[self.hitview==0])

        self.targets_y = torch.zeros((object_target_index_y.amax()+1,), dtype=self.object_targets.dtype) \
            .scatter_(dim=0, index=object_target_index_y, src=self.targets[self.hitview==1])

        view_rate = segment_csr(self.hitview.to(float), indptr=self.hitlim, reduce='mean')
        mask = ((self.targets != -1)).to(float)
        self.mask = (segment_csr(mask, indptr=self.hitlim, reduce='mean') > 0.90) & (view_rate > 0.01) & (view_rate < 0.99)
        #self.mask[test_idxs[(self.min_limit < test_idxs) & (self.max_limit > test_idxs)]] = False
        #self.mask[self.bbox_offsets[1:] == self.bbox_offsets[:-1]] = False
        self.hitlim_lo = hitlim_lo[self.mask].clone()
        self.hitlim_hi = hitlim_hi[self.mask].clone()

        self.bbox_offsets_hi = self.bbox_offsets[1:][self.mask].clone()
        self.bbox_offsets_lo = self.bbox_offsets[:-1][self.mask].clone()

    def compute_statistics(self,
                           mean: Optional[Tensor] = None,
                           std: Optional[Tensor] = None,
                           extra_mean: Optional[Tensor] = None,
                           extra_std: Optional[Tensor] = None,
                           ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if mean is None:
            mean = self.features.mean(0)
            std = self.features.std(0)

            std[std < 1e-5] = 1

        self.mean = mean
        self.std = std

        return mean, std, torch.tensor(0.0), torch.tensor(1.0), None, None
    
    def compute_limit_index(self, limit_index) -> np.ndarray:
        """ Take subsection of the data for training / validation

        Parameters
        ----------
        limit_index : float in [-1, 1], tuple of floats, or array-like
            If a positive float - limit the dataset to the first limit_index percent of the data
            If a negative float - limit the dataset to the last |limit_index| percent of the data
            If a tuple - limit the dataset to [limit_index[0], limit_index[1]] percent of the data
            If array-like or tensor - limit the dataset to the specified indices.

        Returns
        -------
        np.ndarray or torch.Tensor
        """
        # In the float case, we just generate the list with the appropriate bounds
        if isinstance(limit_index, float):
            limit_index = (0.0, limit_index) if limit_index > 0 else (1.0 + limit_index, 1.0)

        # In the list / tuple case, we want a contiguous range
        if isinstance(limit_index, (list, tuple)):
            lower_index = int(round(limit_index[0] * self.num_events))
            upper_index = int(round(limit_index[1] * self.num_events))
            limit_index = np.arange(lower_index, upper_index)

        # Convert to numpy array for simplicity
        if isinstance(limit_index, Tensor):
            limit_index = limit_index.numpy()

        # Make sure the resulting index array is sorted for faster loading.
        return np.sort(limit_index)

    def __len__(self) -> int:
        return len(self.hitlim_lo)

    def __getitem__(self, item) -> List[Tuple[Tensor, ...]]:
        event_low = self.hitlim_lo[item]
        event_high = self.hitlim_hi[item]

        features = torch.Tensor(self.features[event_low:event_high]).clone()
        coordinates = torch.Tensor(self.coordinates[event_low:event_high]).clone()
        object_targets = torch.Tensor(self.object_targets[event_low:event_high]).clone()
        hitview = torch.Tensor(self.hitview[event_low:event_high]).clone()
        hits_index = torch.Tensor(self.id_hits_index[event_low]).clone().item()

        features_x = features[hitview == 0].clone()
        coordinates_x = coordinates[hitview == 0].clone()
        object_targets_x = object_targets[hitview == 0].clone()
        bbox_x = self.bbox_x[self.bbox_offsets_lo[item]:self.bbox_offsets_hi[item]].clone()
        targets_x = self.targets_x[self.bbox_offsets_lo[item]:self.bbox_offsets_hi[item]].clone()

        #print(coordinates_x.shape)
        #print(features_x.shape)

        features_dense_x = torch.sparse_coo_tensor(
            #torch.cat([torch.zeros((coordinates_x.shape[0],1)), coordinates_x], dim=1).T,
            coordinates_x.T,
            features_x.squeeze(1),
            size=(100,80)
        ).to_dense().reshape((1,100,80))
        
        object_targets_dense_x = torch.nn.functional.one_hot(        
            torch.sparse_coo_tensor(
                coordinates_x.T,
                object_targets_x,
                size=(100,80)
            ).to_dense(),
            num_classes=targets_x.shape[0]
        ).transpose(2,1).transpose(0,1).to(torch.uint8)

        features_y = features[hitview == 1].clone()
        coordinates_y = coordinates[hitview == 1].clone()
        object_targets_y = object_targets[hitview == 1].clone()
        bbox_y = self.bbox_y[self.bbox_offsets_lo[item]:self.bbox_offsets_hi[item]].clone()
        targets_y = self.targets_y[self.bbox_offsets_lo[item]:self.bbox_offsets_hi[item]].clone()

        features_dense_y = torch.sparse_coo_tensor(
            #torch.cat([torch.zeros((coordinates_y.shape[0],1)), coordinates_y], dim=1).T,
            coordinates_y.T,
            features_y.squeeze(1),
            size=(100,80)
        ).to_dense().reshape((1,100,80))

        object_targets_dense_y = torch.nn.functional.one_hot(    
            torch.sparse_coo_tensor(
                coordinates_y.T,
                object_targets_y,
                size=(100,80)
            ).to_dense(),
            num_classes=targets_y.shape[0]
        ).transpose(2,1).transpose(0,1).to(torch.uint8)

        #print(features_dense_x.shape)

        return hits_index, (
        features_dense_x,    
        {
            "boxes": bbox_x,
            "labels": targets_x,
            "masks": object_targets_dense_x
        }), (
        features_dense_y,    
        {
            "boxes": bbox_y,
            "labels": targets_y,
            "masks": object_targets_dense_y
        })