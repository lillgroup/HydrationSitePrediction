import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader as PGDataLoader

from hs_prediction.data.io import load_pickle, save_pickle
from hs_prediction.data.pdb import PDB
from hs_prediction.data.utils import get_feature, pool_wrapper, rbf


class HydrationSiteDataset(Dataset):
    """
    Dataset class for hydration site prediction.

    Processes proteins and hydration sites into DGL graphs and labels suitable for training the
    E(3)-equivariant hydration site prediction model.

    Args:
        dataset_path: str
            Path to the dataset. Should be a folder containing one folder per protein system.
        split_path: str
            Path to the split file. Should be a text file with one folder name per line.
        cache_path: str
            Path to the cache folder. Default is "data/cache".
        protein_filename: str
            Name of the protein file. Default is "protein.pdb".
        sites_filename: str
            Name of the hydration sites file. Default is "watsite.csv".
        num_workers: int
            Number of workers to use for preprocessing. Default is 1.
        min_protein_nodes: int, default 10
            Minimum number of nodes in the protein graph.
    """

    def __init__(
        self,
        # transform: Optional[NoiseTransform],
        dataset_path: str,
        split_path: str,
        protein_filename: str = "protein_sasa.pdb",
        num_workers: int = 1,
        occupancy_cutoff: float = 0.5,
        mean=None,
        std=None,
        thermodynamics: bool = False,
        water_type: str = "simulated",
        **_kwargs,
    ):
        super(HydrationSiteDataset, self).__init__(dataset_path)
        self.dataset_path = dataset_path
        self.split_path = split_path
        self.protein_filename = protein_filename
        self.num_workers = num_workers
        self.occupancy_cutoff = occupancy_cutoff
        self.water_type = water_type
        self.mean = mean
        self.std = std
        self.thermodynamics = thermodynamics
        keys = open(split_path, "r").read().strip().split("\n")
        self.keys = keys
        self.cache_name = f"{occupancy_cutoff}".replace(".", "-") + ".pkl"
        self.paths = [
            os.path.join(dataset_path, key, self.cache_name) for key in self.keys
        ]
        self.preprocessing()

    def len(self):
        return len(self.paths)

    def get(self, idx):
        path = self.paths[idx]
        data = load_pickle(path)
        # Apply standardization if mean and std are provided
        if self.thermodynamics:
            if self.mean is not None and self.std is not None:
                data["wat"].enthalpy = (data["wat"].enthalpy - self.mean[0]) / self.std[
                    0
                ]
                data["wat"].entropy = (data["wat"].entropy - self.mean[1]) / self.std[1]
        else:
            del data["wat"].enthalpy
            del data["wat"].entropy
        return data

    def preprocessing(self) -> None:
        """Preprocesses the dataset into DGL graphs and saves it to the cache folder"""
        print(f"Preprocessing proteins from {self.split_path}")
        # Run preprocessing in parallel
        pool_wrapper(
            self.create_data,
            self.keys,
            num_workers=self.num_workers,
            desc="Preprocessing",
        )

    def create_data(self, key: str) -> None:
        """Create a HeteroData object of protein and true hydration sites"""

        output_path = os.path.join(self.dataset_path, key, self.cache_name)
        # if not os.path.exists(output_path):
        if True:
            # Load protein
            protein = PDB.from_file(
                os.path.join(self.dataset_path, key, self.protein_filename)
            )
            protein = PDB(protein[protein["sym"] != "H"])
            protein_coords = protein.get_coords()
            one_hot_atom_types = np.vstack(
                [get_feature("ATOM_TYPE", v, safe=True) for v in protein["name"].values]
            )
            one_hot_atom_residues = np.vstack(
                [get_feature("RESIDUE", v, safe=True) for v in protein["resn"].values]
            )
            rbf_expansion_sasa = rbf(
                protein["fac"].values,  # SASA values?
                min_val=0.0,
                max_val=40.0,
                n_kernels=32,
                gamma=0.1,
            )
            # add zeroes column to allow to distinguish from one hot encoding of water nodes:
            zeros_column = np.zeros((len(rbf_expansion_sasa), 1))
            protein_features = np.hstack(
                [
                    one_hot_atom_types,  # 38 features
                    one_hot_atom_residues,  # 21 features
                    rbf_expansion_sasa,  # 32 features
                    zeros_column,  # 1 feature for one hot encoding if water node or not
                ]
            )
            # Create hetero-graph
            data = HeteroData()
            data["key"] = key
            data["pro"].features = torch.tensor(protein_features, dtype=torch.float32)
            data["pro"].pos = torch.tensor(protein_coords, dtype=torch.float32)
            sasa_cutoff = 0.1
            data["pro"].put_wat = torch.tensor(protein["fac"] > sasa_cutoff)
            # Load hydration sites
            if self.water_type == "simulated":
                sites = pd.read_csv(os.path.join(self.dataset_path, key, "watsite.csv"))
                water_mask = sites["occupancy"] > self.occupancy_cutoff
                sites = sites[water_mask]
                sites_coords = sites[["center_x", "center_y", "center_z"]].values
                data["wat"].pos = torch.tensor(sites_coords, dtype=torch.float32)
                data["wat"].occupancy = torch.tensor(
                    sites["occupancy"], dtype=torch.float32
                )
                data["wat"].enthalpy = torch.tensor(
                    sites["enthalpy"], dtype=torch.float32
                )
                data["wat"].entropy = torch.tensor(
                    sites["entropy"], dtype=torch.float32
                )
            elif self.water_type == "experimental":
                sites = pd.read_csv(
                    os.path.join(self.dataset_path, key, "water_coordinates.csv")
                )
                sites_coords = sites[["center_x", "center_y", "center_z"]].values
                data["wat"].pos = torch.tensor(sites_coords, dtype=torch.float32)
                data["wat"].occupancy = torch.ones(data["wat"].pos.shape[0])
            save_pickle(output_path, data)

    def compute_mean_std(self):
        enthalpies = []
        entropies = []
        for path in self.paths:
            data = load_pickle(path)
            enthalpies.append(data["wat"].enthalpy)
            entropies.append(data["wat"].entropy)
        enthalpies = torch.cat(enthalpies)
        entropies = torch.cat(entropies)
        mean_enthalpy = enthalpies.mean()
        std_enthalpy = enthalpies.std()
        mean_entropy = entropies.mean()
        std_entropy = entropies.std()
        self.mean = (mean_enthalpy, mean_entropy)
        self.std = (std_enthalpy, std_entropy)


def create_dataloaders(config: DictConfig):
    dataset_kwargs = {
        "dataset_path": config.data.dataset_path,
        "num_workers": config.training.num_workers,
        "water_type": config.data.water_type,
        "thermodynamics": config.data.thermodynamics,
        "occupancy_cutoff": config.data.occupancy_cutoff,
    }
    train_dataset = HydrationSiteDataset(
        split_path=config.data.train_split_path, **dataset_kwargs
    )
    if config.data.thermodynamics and config.data.standardize:
        train_dataset.compute_mean_std()  # Compute mean and std for standardization
    valid_dataset = HydrationSiteDataset(
        split_path=config.data.valid_split_path,
        mean=train_dataset.mean,
        std=train_dataset.std,
        **dataset_kwargs,
    )
    train_loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }
    train_dataloader = PGDataLoader(
        dataset=train_dataset, shuffle=True, **train_loader_kwargs
    )
    valid_loader_kwargs = {
        "batch_size": config.inference.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }
    valid_dataloader = PGDataLoader(
        dataset=valid_dataset, shuffle=False, **valid_loader_kwargs
    )
    return train_dataloader, valid_dataloader, train_dataset.mean, train_dataset.std


def create_inference_dataloader(config: DictConfig):
    dataset_kwargs = {
        "dataset_path": config.data.dataset_path,
        "num_workers": config.training.num_workers,
        "water_type": config.data.water_type,
        "thermodynamics": config.data.thermodynamics,
        "transform": None,
    }
    dataloader_kwargs = {
        "batch_size": config.inference.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }

    dataset = HydrationSiteDataset(
        split_path=config.data.test_split_path, **dataset_kwargs
    )
    dataloader = PGDataLoader(dataset=dataset, shuffle=False, **dataloader_kwargs)
    return dataloader
