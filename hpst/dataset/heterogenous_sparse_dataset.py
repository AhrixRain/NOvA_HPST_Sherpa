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

class HeterogenousSparseDataset(Dataset):
    def __init__(self, data_file: str, limit_index=1.0,  transform=None, testing=False):
        super(HeterogenousSparseDataset, self).__init__()

        self.pixel_mean = 0
        self.pixel_std = 1
        self.pixels = None

        # Needed to make compatible with other datasets
        self.mean = 0
        self.std = 1

        self.extra_mean = 0
        self.extra_std = 1

        #test_idxs = torch.load("/home/roblesee/dune/UCI-NuML-Codebase/NOvA/Transformer/test_idxs.pt")

        print("loading file")
        with h5py.File(data_file, 'r') as file:
            # the masking operation in h5py is abysmally slow
            # under no circumstances try to do numpy operations
            # directly to the h5py file, instead load the numpy
            # array into memory and do that instead
            print("1")
            hits_index = file["cvnmap_index"][:,0]
            print("1.1")
            self.num_events = hits_index.max()
            # hitlim_mask = np.nonzero(nhit)
            # masked_hits = nhit[hitlim_mask]
            # cumsum_hits = np.cumsum(masked_hits)
            # self.hitlim = np.pad(cumsum_hits, (1,0), 'constant')
    
            print("2")
            hit_differences = torch.argwhere(torch.from_numpy(hits_index[1:] != hits_index[:-1]))[:,0]+1
            self.hitlim = np.pad(hit_differences, (1,0), 'constant')
            limit_index = self.compute_limit_index(limit_index)
            self.min_limit = limit_index.min()
            self.max_limit = limit_index.max()
            print("3")
            self.hits_index = torch.from_numpy(file["cvnmap_index"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit], 0].astype(np.int64))
            self.features = torch.from_numpy(file["cvnmap_value"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit]].astype(np.float32).reshape((-1,1)))
            self.targets = torch.from_numpy(file["cvnmap_label"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit]].astype(np.int64)).to(int)
            self.object_targets = torch.from_numpy(file["cvnmap_object"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit]].astype(np.int64)).to(int) - 1
            coords = torch.from_numpy(file["cvnmap_index"][self.hitlim[self.min_limit]:self.hitlim[self.max_limit], 1].astype(np.float32))
            img_width, img_height = 100, 80
            self.hitview = coords // (img_width * img_height)
            X = (coords % (img_width*img_height)) // img_height
            YZ = coords % img_height
            print("4")
            self.coordinates = torch.stack([X, YZ], dim=-1)
            #test_idxs -= self.hitlim[self.min_limit]
            self.hitlim = torch.from_numpy((self.hitlim[self.min_limit:self.max_limit] - self.hitlim[self.min_limit]).astype(int))
            hitlim_lo = self.hitlim[:-1]
            hitlim_hi = self.hitlim[1:]

        print("processing file")
    
        self.num_features = self.features.shape[-1]
        self.coord_dim = self.coordinates.shape[-1]
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
                4: -1, # neutron -> ignore
                5: 5, # Pion -> Pion
                6: -1, # PiZero/Neutral Pion -> ignore
                7: 4, # Gamma/Photon -> Photon
                8: 3, # MostlyHA -> proton
                9: 2, # MostlyEM -> electron
                10: 1, # MostlyMU -> muon
                11: -1 # Unknown -> ignore
            }
            targets_copy = self.targets.clone()

            for key, value in mapping.items():
                self.targets[targets_copy == key] = value
            self.target_count = torch.bincount(self.targets[self.targets != -1].to(int))
            self.num_classes = 6


        view_rate = segment_csr(self.hitview.to(float), indptr=self.hitlim, reduce='mean')
        mask = ((self.targets != -1)).to(float)
        self.mask = (segment_csr(mask, indptr=self.hitlim, reduce='mean') > 0.90) & (view_rate > 0.01) & (view_rate < 0.99)
        """
        if testing:
            test_mask = torch.ones_like(self.mask)
            test_mask[test_idxs[(self.min_limit < test_idxs) & (self.max_limit > test_idxs)]] = False
            self.mask = self.mask & ~test_mask
        else:
            self.mask[test_idxs[(self.min_limit < test_idxs) & (self.max_limit > test_idxs)]] = False
        """
        self.hitlim_lo = hitlim_lo[self.mask].clone()
        self.hitlim_hi = hitlim_hi[self.mask].clone()

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
        targets = torch.Tensor(self.targets[event_low:event_high]).clone()
        object_targets = torch.Tensor(self.object_targets[event_low:event_high]).clone()
        hitview = torch.Tensor(self.hitview[event_low:event_high]).clone()
        hits_index = torch.Tensor(self.hits_index[event_low]).clone().item()

        if self.transform is not None:
            data = Data(x=features, pos=coordinates, y=targets)
            data = self.transform(data)

            features = data.x
            coordinates = data.pos
            targets = data.y

        features_x = features[hitview == 0].clone()
        coordinates_x = coordinates[hitview == 0].clone()
        targets_x = targets[hitview == 0].clone()
        object_targets_x = object_targets[hitview == 0].clone()

        features_y = features[hitview == 1].clone()
        coordinates_y = coordinates[hitview == 1].clone()
        targets_y = targets[hitview == 1].clone()
        object_targets_y = object_targets[hitview == 1].clone()

        return hits_index, (
            features_x,
            coordinates_x,
            targets_x,
            object_targets_x
        ), (
            features_y,
            coordinates_y,
            targets_y,
            object_targets_y
        )