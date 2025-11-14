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
from pathlib import Path
import pandas as pd

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

class BboxSimMaskDataset(Dataset):
    def __init__(self, data_file: str, limit_index=1.0,  transform=None, testing=False):
        super(BboxSimMaskDataset, self).__init__()
        self.pixel_mean = 0
        self.pixel_std = 1
        self.pixels = None

        # Needed to make compatible with other datasets
        self.mean = 0
        self.std = 1

        self.extra_mean = 0
        self.extra_std = 1

        #test_idxs = torch.load("/home/roblesee/dune/UCI-NuML-Codebase/NOvA/Transformer/test_idxs.pt")
        BASE_DIR = Path(data_file)
        CACHE_DIR = (BASE_DIR / ".." / f"{BASE_DIR.name}_cache").resolve()
        CACHE_FILE = CACHE_DIR / f"cache_{limit_index[0]}_{limit_index[1]}.pt"

        if CACHE_DIR.exists() and CACHE_FILE.exists():
            cache = torch.load(CACHE_FILE)

            self.hits_index = cache["hits_index"]
            self.features = cache["features"]
            self.targets = cache["targets"]
            self.object_targets = cache["object_targets"]
            self.coordinates = cache["coordinates"]
            self.num_features = cache["num_features"]
            self.coord_dim = cache["coord_dim"]
            self.mask = cache["mask"]
            self.hitlim_lo = cache["hitlim_lo"]
            self.hitlim_hi = cache["hitlim_hi"]
        else:
            NUMU_DIRECTORY = BASE_DIR / "argon_cubic_production"
            NUE_DIRECTORY = BASE_DIR / "argon_cubic_production"

            numu_files = list(NUMU_DIRECTORY.glob("argon_cubic_numu_production_*.h5"))
            nue_files = list(NUE_DIRECTORY.glob("argon_cubic_nue_production_*.h5"))

            numu_limit = (int(len(numu_files)*limit_index[0]),int(len(numu_files)*limit_index[1]))
            nue_limit = (int(len(numu_files)*limit_index[0]),int(len(nue_files)*limit_index[1]))
            files_list = (numu_files[numu_limit[0]:numu_limit[1]] + nue_files[nue_limit[0]:nue_limit[1]])

            class_names = {
                "Other": np.inf,
                "Electron": 11,
                "Muon": 13,
                "Proton": 2212,
                "Neutron": 2112,
                "Charged Pion": 211,
                "Neutral Pion": 111,
                "Photon": 22
            }

            class_ids = np.asarray(list(class_names.values()))

            print("loading files")
            current_eid = 0
            classess = []
            prongss = []
            edeps = []
            coordss = []
            evtids = []
            for data_file in tqdm(files_list):
                with h5py.File(data_file, 'r') as file:
                    cross_comp = file["g4_data_0"]["pdg"][:] == np.expand_dims(class_ids, axis=0)
                    mask = cross_comp.any(axis=1)
                    classes = cross_comp.argmax(axis=1)

                    coords = np.stack([file["g4_data_0"]["step_x"][:,0], file["g4_data_0"]["step_y"][:,0], file["g4_data_0"]["step_z"][:,0]], axis=-1)
                    evtid = file["g4_data_0"]["evtid"][:,0]
                    pid = file["g4_data_0"]["pid"][:,0]
                    prongs = file["g4_data_0"]["tid"][:,0]
                    edep = file["g4_data_0"]["step_edep"][:,0]
                    t = file["g4_data_0"]["step_no"][:,0]


                    if True:
                        for i in range(evtid.max() + 1):
                            mask = evtid == i

                            prongsum = np.zeros((np.unique(prongs[mask]).max()+1,))
                            np.add.at(prongsum, prongs[mask], pid[mask])

                            prongcount = np.zeros((np.unique(prongs[mask]).max()+1,))
                            np.add.at(prongcount, prongs[mask], np.ones_like(pid[mask]))

                            prongcount[prongcount == 0] = 1
                            parents = (prongsum/prongcount).astype(int)

                            classsum = np.zeros((np.unique(prongs[mask]).max()+1,))
                            np.add.at(classsum, prongs[mask], classes[mask])

                            parentclasses = (classsum/prongcount).astype(int)

                            recursiveparent = parents.copy()

                            while not (parents[recursiveparent] == 0).all():
                                # take a step
                                step = parents[recursiveparent]
                                
                                # only save when parent isn't a primary node
                                stepmask = step != 0
                                recursiveparent[stepmask] = step[stepmask]
                            
                            primary_prongs = np.unique(recursiveparent)

                            # remove primary prongs
                            mask[mask] =  ~np.isin(prongs[mask], primary_prongs)

                            prongs[mask] = recursiveparent[prongs[mask]]
                            
                            classes[mask] = parentclasses[prongs[mask]]

                    #mask = (edep > 0.25)
                    mask = (edep > 0.05)
                    #mask = ...
            
                    t = t[mask]
                    classes = classes[mask]
                    coords = coords[mask]
                    prongs = prongs[mask]
                    edep = edep[mask]
                    evtid = evtid[mask]
                    pid = pid[mask]

                    # TODO: don't use pandas as I think it's kind of slow, maybe pytorch is faster
                    coords = (coords // np.asarray([[50,50,50]])).astype(np.int64)
                    df = pd.DataFrame({
                        "evtid": evtid,
                        "x": coords[:,0], 
                        "y": coords[:,1], 
                        "z": coords[:,2],
                        "c": classes, 
                        "prong": prongs, 
                        "E": edep
                    })
                    df_x = df.groupby(by=["evtid","x","y","z"]).agg({
                        "c": lambda x: pd.Series.mode(x)[0],
                        "prong": lambda x: pd.Series.mode(x)[0],
                        "E": "sum"
                    }).reset_index()
                    classes = torch.from_numpy(df_x["c"].to_numpy())
                    prongs = torch.from_numpy(df_x["prong"].to_numpy())
                    edep = torch.from_numpy(df_x["E"].to_numpy())
                    coords = torch.from_numpy(df_x[["x","y","z"]].to_numpy())

                    evtid = torch.from_numpy(df_x["evtid"].to_numpy() + current_eid)

                    current_eid = evtid.amax().item() + 1

                    classess.append(classes)
                    prongss.append(prongs)
                    edeps.append(edep)
                    coordss.append(coords)
                    evtids.append(evtid)

            
            self.hits_index = torch.concat(evtids, dim=0)
            self.features = torch.concat(edeps, dim=0).unsqueeze(1)
            self.targets = torch.concat(classess, dim=0)
            self.object_targets = torch.concat(prongss, dim=0)
            self.coordinates = torch.concat(coordss, dim=0).to(torch.float32)

            hit_differences = torch.argwhere((self.hits_index[1:] != self.hits_index[:-1]))[:,0]+1
            self.hitlim = torch.from_numpy(np.pad(hit_differences, (1,0), 'constant'))
            hitlim_lo = self.hitlim[:-1]
            hitlim_hi = self.hitlim[1:]

            print("processing file")
        
            self.num_features = self.features.shape[-1]
            self.coord_dim = self.coordinates.shape[-1]

            #view_rate = segment_csr(self.hitview.to(float), indptr=self.hitlim, reduce='mean')
            mask = ((self.targets != -1)).to(float)
            self.mask = (segment_csr(mask, indptr=self.hitlim, reduce='mean') > 0.90)# & (view_rate > 0.01) & (view_rate < 0.99)
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

            
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "hits_index": self.hits_index,
                "features": self.features,
                "targets": self.targets,
                "object_targets": self.object_targets,
                "coordinates": self.coordinates,
                "num_features": self.num_features,
                "coord_dim": self.coord_dim,
                "mask": self.mask,
                "hitlim_lo": self.hitlim_lo,
                "hitlim_hi": self.hitlim_hi
            }, CACHE_FILE)



        self.num_classes = 8
        self.transform = transform
        print(self.targets.shape)
        print(self.targets.min(), self.targets.max())
        self.target_count = torch.bincount(self.targets[(self.targets != -1)].to(int), minlength=self.num_classes) + 1
        print(self.target_count)
        # x \in (-2000, 2000), y \in (-2000, 2000), z \in (0, 7000)
        self.resolution = np.asarray([4000, 4000, 7000]) // np.asarray([50, 50, 50])
        self.coord_mins = np.asarray([-2000, -2000, 0]) // np.asarray([50, 50, 50])

        self.coordinates = self.coordinates - self.coord_mins

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

    def __getitem__(self, item) -> Tuple[int, Tensor, Tensor, Tensor]:
        event_low = self.hitlim_lo[item]
        event_high = self.hitlim_hi[item]

        features = torch.Tensor(self.features[event_low:event_high]).clone()
        coordinates = torch.Tensor(self.coordinates[event_low:event_high]).clone()
        object_targets = torch.Tensor(self.object_targets[event_low:event_high]).clone()
        hits_index = torch.Tensor(self.hits_index[event_low]).clone().item()
        targets = self.targets[event_low:event_high].clone()

        features_dense = torch.sparse_coo_tensor(
            #torch.cat([torch.zeros((coordinates_x.shape[0],1)), coordinates_x], dim=1).T,
            coordinates.T,
            features.squeeze(1),
            size=tuple(self.resolution)
        ).to_dense().unsqueeze(0)
        
        #object_targets_dense = torch.nn.functional.one_hot(        
        object_targets_dense = torch.sparse_coo_tensor(
            coordinates.T,
            object_targets,
            size=tuple(self.resolution)
        ).to_dense().to(torch.int64)
        #    num_classes=32
        #).transpose(3,2).transpose(2,1).transpose(0,1).to(torch.uint8)

        #targets_dense = torch.nn.functional.one_hot(    
        targets_dense = torch.sparse_coo_tensor(
            coordinates.T,
            targets,
            size=tuple(self.resolution)
        ).to_dense().to(torch.int64)
        #    num_classes=8
        #).transpose(3,2).transpose(2,1).transpose(0,1).to(torch.uint8)

        #print(features_dense_x.shape)

        return hits_index, features_dense, targets_dense, object_targets_dense
        