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

from pathlib import Path
import pandas as pd

from torch_geometric.nn.pool import radius
from torch_geometric.utils import scatter

def batch2offset(batch, minlength=0):
    return torch.cumsum(batch.bincount(minlength=minlength), dim=0).long()

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
        print(data_file)
        BASE_DIR = Path(data_file)
        CACHE_DIR = (BASE_DIR / ".." / f"{BASE_DIR.name}_cache").resolve()
        CACHE_FILE = CACHE_DIR / f"2d_cache_{limit_index[0]}_{limit_index[1]}.pt"

        if CACHE_DIR.exists() and CACHE_FILE.exists():
            cache = torch.load(CACHE_FILE)

            self.hits_index_x = cache["hits_index_x"]
            self.features_x = cache["features_x"]
            self.targets_x = cache["targets_x"]
            self.object_targets_x = cache["object_targets_x"]
            self.coordinates_x = cache["coordinates_x"]
            self.mask_x = cache["mask_x"]
            self.hitlim_lo_x = cache["hitlim_lo_x"]
            self.hitlim_hi_x = cache["hitlim_hi_x"]

            self.hits_index_y = cache["hits_index_y"]
            self.features_y = cache["features_y"]
            self.targets_y = cache["targets_y"]
            self.object_targets_y = cache["object_targets_y"]
            self.coordinates_y = cache["coordinates_y"]
            self.mask_y = cache["mask_y"]
            self.hitlim_lo_y = cache["hitlim_lo_y"]
            self.hitlim_hi_y = cache["hitlim_hi_y"]
            
            self.num_features = cache["num_features"]
            self.coord_dim = cache["coord_dim"]
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
                "Kaon": 321,
                #"Photon": 22,
                #"Electron-": -11,
                #"Muon-": -13
                #"Pion-": -211,
                #"Kaon-": -321,
            }

            class_ids = np.asarray(list(class_names.values()))

            print("loading files")
            current_eid = 0
            classess_x = []
            prongss_x = []
            edeps_x = []
            coordss_x = []
            evtids_x = []

            classess_y = []
            prongss_y = []
            edeps_y = []
            coordss_y = []
            evtids_y = []
            for data_file in tqdm(files_list):
                with h5py.File(data_file, 'r') as file:
                    pdgs = file["g4_data_0"]["pdg"][:]
                    pdgs[pdgs == -11] = 11
                    pdgs[pdgs == -13] = 13
                    pdgs[pdgs == -211] = 211
                    pdgs[pdgs == -321] = 321
                    cross_comp = pdgs == np.expand_dims(class_ids, axis=0)
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

                            #print("pid", pid[mask], "prongs", prongs[mask])

                            """
                            prongsum = np.zeros((np.unique(prongs[mask]).max()+1,))
                            np.add.at(prongsum, prongs[mask], pid[mask])

                            prongcount = np.zeros((np.unique(prongs[mask]).max()+1,))
                            np.add.at(prongcount, prongs[mask], np.ones_like(pid[mask]))
                            
                            print(prongsum, prongcount)

                            prongcount[prongcount == 0] = 1
                            parents = (prongsum/prongcount).astype(int)
                            print(parents)
                            """

                            parents = np.zeros((np.unique(prongs[mask]).max()+1,))
                            parents[prongs[mask]] = pid[mask]
                            #print(parents)

                            # parents[prongs[mask]] = pid[mask]

                            #classsum = np.zeros((np.unique(prongs[mask]).max()+1,))
                            #np.add.at(classsum, prongs[mask], classes[mask])

                            #parentclasses = (classsum/prongcount).astype(int)

                            parentclasses = np.zeros((np.unique(prongs[mask]).max()+1,))
                            parentclasses[prongs[mask]] = classes[mask]

                            parents = parents.astype(int)
                            parentclasses = parentclasses.astype(int)

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

                            # relabel the prongs so that they're in a normal range
                            _, prongs[evtid == i] = np.unique(prongs[evtid == i], return_inverse=True)

                        # print(prongs.min(), prongs.max())

                    #mask = (edep > 0.25)
                    #mask = (edep > 0.05)
                    mask = ...
            
                    t = t[mask]
                    classes = classes[mask]
                    coords = coords[mask]
                    prongs = prongs[mask]
                    edep = edep[mask]
                    evtid = evtid[mask]
                    pid = pid[mask]

                    #print(coords.min(axis=0), coords.max(axis=0))
                    # TODO: don't use pandas as I think it's kind of slow, maybe pytorch is faster
                    """
                    coords = (coords // np.asarray([[50,50,50]])).astype(np.int64)
                    zmask = (coords[:,2] % 2 == 0)
                    
                    df = pd.DataFrame({
                        "evtid": evtid[:],
                        "x": coords[:,0], 
                        "y": coords[:,1], 
                        "z": coords[:,2],
                        "c": classes[:], 
                        "prong": prongs[:], 
                        "E": edep[:]
                    })
                    df_x = df[zmask].groupby(by=["evtid","x","z"]).agg({
                        "c": lambda x: pd.Series.mode(x)[0],
                        "prong": lambda x: pd.Series.mode(x)[0],
                        "E": "sum"
                    }).reset_index()
                    classes_x = torch.from_numpy(df_x["c"].to_numpy())
                    prongs_x = torch.from_numpy(df_x["prong"].to_numpy())
                    edep_x = torch.from_numpy(df_x["E"].to_numpy())
                    coords_x = torch.from_numpy(df_x[["x","z"]].to_numpy())
                    evtid_x = torch.from_numpy(df_x["evtid"].to_numpy() + current_eid)

                    df_y = df[~zmask].groupby(by=["evtid","y","z"]).agg({
                        "c": lambda x: pd.Series.mode(x)[0],
                        "prong": lambda x: pd.Series.mode(x)[0],
                        "E": "sum"
                    }).reset_index()
                    classes_y = torch.from_numpy(df_y["c"].to_numpy())
                    prongs_y = torch.from_numpy(df_y["prong"].to_numpy())
                    edep_y = torch.from_numpy(df_y["E"].to_numpy())
                    coords_y = torch.from_numpy(df_y[["y","z"]].to_numpy())
                    evtid_y = torch.from_numpy(df_y["evtid"].to_numpy() + current_eid)
                    """
                    coords = (coords // np.asarray([[5,5,5]])).astype(np.int64)

                    
                    index_x = np.stack([evtid, coords[:,0], coords[:,2]], axis=0)
                    unique_index_x, inverse_index_x = np.unique(index_x, axis=1, return_inverse=True)

                    write_order = np.argsort(edep)

                    edepa_x = np.zeros((unique_index_x.shape[-1],), dtype=edep.dtype)
                    np.add.at(edepa_x, inverse_index_x, edep)
                    edep_x = torch.from_numpy(edepa_x)

                    classesa_x = np.zeros((unique_index_x.shape[-1],))
                    classesa_x[inverse_index_x[write_order]] = classes[write_order]
                    classes_x = torch.from_numpy(classesa_x).to(torch.int64)

                    prongsa_x = np.zeros((unique_index_x.shape[-1],))
                    prongsa_x[inverse_index_x[write_order]] = prongs[write_order]
                    prongs_x = torch.from_numpy(prongsa_x).to(torch.int64)

                    evtid_x = torch.from_numpy(unique_index_x[0]) + current_eid
                    coords_x = torch.from_numpy(unique_index_x[1:3].T)
                    
                    graph = radius(coords_x, coords_x, r=1.8, batch_x=evtid_x, batch_y=evtid_x)

                    neighbor_energy_x = scatter(edep_x[graph[1]], graph[0], dim=0, reduce='sum', dim_size=edep_x.shape[0]) - edep_x
                    
                    mask_x = (edep_x > 0.2) & (neighbor_energy_x > 0.05) & (coords_x[:,1] % 2 == 0)                    

                    classess_x.append(classes_x[mask_x])
                    prongss_x.append(prongs_x[mask_x])
                    edeps_x.append(edep_x[mask_x])
                    coordss_x.append(coords_x[mask_x])
                    evtids_x.append(evtid_x[mask_x])

                    index_y = np.stack([evtid, coords[:,1], coords[:,2]], axis=0)
                    unique_index_y, inverse_index_y = np.unique(index_y, axis=1, return_inverse=True)

                    edepa_y = np.zeros((unique_index_y.shape[-1],), dtype=edep.dtype)
                    np.add.at(edepa_y, inverse_index_y, edep)
                    edep_y = torch.from_numpy(edepa_y)

                    classesa_y = np.zeros((unique_index_y.shape[-1],))
                    classesa_y[inverse_index_y[write_order]] = classes[write_order]
                    classes_y = torch.from_numpy(classesa_y).to(torch.int64)

                    prongsa_y = np.zeros((unique_index_y.shape[-1],))
                    prongsa_y[inverse_index_y[write_order]] = prongs[write_order]
                    prongs_y = torch.from_numpy(prongsa_y).to(torch.int64)

                    evtid_y = torch.from_numpy(unique_index_y[0]) + current_eid
                    coords_y = torch.from_numpy(unique_index_y[1:3].T)

                    graph = radius(coords_y, coords_y, r=1.8, batch_x=evtid_y, batch_y=evtid_y)

                    neighbor_energy_y = scatter(edep_y[graph[1]], graph[0], dim=0, reduce='sum', dim_size=edep_y.shape[0]) - edep_y
                    
                    mask_y = (edep_y > 0.2) & (neighbor_energy_y > 0.05) & (coords_y[:,1] % 2 == 1)                    

                    classess_y.append(classes_y[mask_y])
                    prongss_y.append(prongs_y[mask_y])
                    edeps_y.append(edep_y[mask_y])
                    coordss_y.append(coords_y[mask_y])
                    evtids_y.append(evtid_y[mask_y])

                    current_eid = max(evtid_x[mask_x].amax().item(), evtid_y[mask_y].amax().item()) + 1

            
            self.hits_index_x = torch.concat(evtids_x, dim=0).to(torch.int64)
            self.features_x = torch.concat(edeps_x, dim=0).unsqueeze(1)
            self.targets_x = torch.concat(classess_x, dim=0)
            self.object_targets_x = torch.concat(prongss_x, dim=0)
            self.coordinates_x = torch.concat(coordss_x, dim=0).to(torch.float32)

            self.hits_index_y = torch.concat(evtids_y, dim=0)
            self.features_y = torch.concat(edeps_y, dim=0).unsqueeze(1)
            self.targets_y = torch.concat(classess_y, dim=0)
            self.object_targets_y = torch.concat(prongss_y, dim=0)
            self.coordinates_y = torch.concat(coordss_y, dim=0).to(torch.float32)


            minlength = max(self.hits_index_x.amax(), self.hits_index_y.amax())
            hit_differences_x = torch.argwhere((self.hits_index_x[1:] != self.hits_index_x[:-1]))[:,0]+1
            #self.hitlim_x = torch.from_numpy(np.pad(hit_differences_x, (1,1), 'constant', constant_values=(0,-1)))
            #old_hitlim_x = torch.from_numpy(np.pad(hit_differences_x, (1,1), 'constant', constant_values=(0,-1)))
            self.hitlim_x = batch2offset(self.hits_index_x, minlength=minlength)
            hitlim_lo_x = self.hitlim_x[:-1]
            hitlim_hi_x = self.hitlim_x[1:]

            #hit_differences_y = torch.argwhere((self.hits_index_y[1:] != self.hits_index_y[:-1]))[:,0]+1
            #self.hitlim_y = torch.from_numpy(np.pad(hit_differences_y, (1,1), 'constant', constant_values=(0,-1)))
            self.hitlim_y = batch2offset(self.hits_index_y, minlength=minlength)
            hitlim_lo_y = self.hitlim_y[:-1]
            hitlim_hi_y = self.hitlim_y[1:]

            print("processing file")
        
            self.num_features = self.features_x.shape[-1]
            self.coord_dim = self.coordinates_x.shape[-1]

            #view_rate = segment_csr(self.hitview.to(float), indptr=self.hitlim, reduce='mean')
            mask_x = ((self.targets_x != -1)).to(float)
            self.mask_x = (segment_csr(mask_x, indptr=self.hitlim_x, reduce='mean') > 0.90)# & (view_rate > 0.01) & (view_rate < 0.99)
            mask_y = ((self.targets_y != -1)).to(float)
            self.mask_y = (segment_csr(mask_y, indptr=self.hitlim_y, reduce='mean') > 0.90)# & (view_rate > 0.01) & (view_rate < 0.99)

            """
            if testing:
                test_mask = torch.ones_like(self.mask)
                test_mask[test_idxs[(self.min_limit < test_idxs) & (self.max_limit > test_idxs)]] = False
                self.mask = self.mask & ~test_mask
            else:
                self.mask[test_idxs[(self.min_limit < test_idxs) & (self.max_limit > test_idxs)]] = False
            """
            self.hitlim_lo_x = hitlim_lo_x[self.mask_x & self.mask_y].clone()
            self.hitlim_hi_x = hitlim_hi_x[self.mask_x & self.mask_y].clone()
            self.hitlim_lo_y = hitlim_lo_y[self.mask_x & self.mask_y].clone()
            self.hitlim_hi_y = hitlim_hi_y[self.mask_x & self.mask_y].clone()

            
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "hits_index_x": self.hits_index_x,
                "features_x": self.features_x,
                "targets_x": self.targets_x,
                "object_targets_x": self.object_targets_x,
                "coordinates_x": self.coordinates_x,
                "mask_x": self.mask_x,
                "hitlim_lo_x": self.hitlim_lo_x,
                "hitlim_hi_x": self.hitlim_hi_x,
                "hits_index_y": self.hits_index_y,
                "features_y": self.features_y,
                "targets_y": self.targets_y,
                "object_targets_y": self.object_targets_y,
                "coordinates_y": self.coordinates_y,
                "mask_y": self.mask_y,
                "hitlim_lo_y": self.hitlim_lo_y,
                "hitlim_hi_y": self.hitlim_hi_y,
                "num_features": self.num_features,
                "coord_dim": self.coord_dim,
            }, CACHE_FILE)

        self.num_classes = 8
        self.transform = transform
        print(self.targets_x.shape)
        print(self.targets_x.min(), self.targets_x.max())
        self.target_count = torch.bincount(self.targets_x[(self.targets_x != -1)].to(int), minlength=self.num_classes) + 1
        print(self.target_count)

    def compute_statistics(self,
                           mean: Optional[Tensor] = None,
                           std: Optional[Tensor] = None,
                           extra_mean: Optional[Tensor] = None,
                           extra_std: Optional[Tensor] = None,
                           ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if mean is None:
            mean_x = self.features_x.mean(0)
            std_x = self.features_x.std(0)

            std_x[std_x < 1e-5] = 1

            mean_y = self.features_y.mean(0)
            std_y = self.features_y.std(0)

            std_y[std_y < 1e-5] = 1

        self.mean_x = mean_x
        self.std_x = std_x

        self.mean_y = mean_y
        self.std_y = std_y

        return mean_x, std_x, torch.tensor(0.0), torch.tensor(1.0), None, None

    def __len__(self) -> int:
        return len(self.hitlim_lo_x)

    def __getitem__(self, item) -> List[Tuple[Tensor, ...]]:
        event_low_x = self.hitlim_lo_x[item]
        event_high_x = self.hitlim_hi_x[item]

        features_x = torch.Tensor(self.features_x[event_low_x:event_high_x]).clone()
        coordinates_x = torch.Tensor(self.coordinates_x[event_low_x:event_high_x]).clone()
        targets_x = torch.Tensor(self.targets_x[event_low_x:event_high_x]).clone()
        object_targets_x = torch.Tensor(self.object_targets_x[event_low_x:event_high_x]).clone()
        hits_index_x = torch.Tensor(self.hits_index_x[event_low_x]).clone().item()

        event_low_y = self.hitlim_lo_y[item]
        event_high_y = self.hitlim_hi_y[item]

        features_y = torch.Tensor(self.features_y[event_low_y:event_high_y]).clone()
        coordinates_y = torch.Tensor(self.coordinates_y[event_low_y:event_high_y]).clone()
        targets_y = torch.Tensor(self.targets_y[event_low_y:event_high_y]).clone()
        object_targets_y = torch.Tensor(self.object_targets_y[event_low_y:event_high_y]).clone()
        hits_index_y = torch.Tensor(self.hits_index_y[event_low_y]).clone().item()

        _, object_targets = torch.unique(torch.cat([object_targets_x, object_targets_y], dim=0), return_inverse=True)
        object_targets_x, object_targets_y = object_targets[:object_targets_x.shape[0]], object_targets[object_targets_x.shape[0]:]

        if self.transform is not None:
            data = Data(x=features_x, pos=coordinates_x, y=targets_x)
            data = self.transform(data)

            features_x = data.x
            coordinates_x = data.pos
            targets_x = data.y

            data = Data(x=features_y, pos=coordinates_y, y=targets_y)
            data = self.transform(data)

            features_y = data.x
            coordinates_y = data.pos
            targets_y = data.y

        features_x = features_x.clone()
        coordinates_x = coordinates_x.clone()
        targets_x = targets_x.clone()
        object_targets_x = object_targets_x.clone()

        features_y = features_y.clone()
        coordinates_y = coordinates_y.clone()
        targets_y = targets_y.clone()
        object_targets_y = object_targets_y.clone()

        #assert torch.isfinite(features_x).all()
        #assert torch.isfinite(coordinates_x).all()
        #assert torch.isfinite(targets_x).all()
        #assert torch.isfinite(object_targets_x).all()

        return hits_index_x, (
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