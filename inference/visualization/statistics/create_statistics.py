"""Create statistics about the dataset."""

import os

import hydra
import matplotlib.pyplot as plt
import torch

from hs_prediction.data.dataset import create_dataloaders


def plot_frequency(element_list, label, title, save_path, color, bins=20):
    fig, ax = plt.subplots()
    ax.hist(element_list, bins=bins, alpha=0.5, label=label, color=color)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    plt.savefig(save_path)
    plt.close(fig)


def num_water_statistics(config):
    train_dataloader, valid_dataloader, _, _ = create_dataloaders(config)
    enthalpy = []
    entropy = []
    base_path = os.path.join(config.general.repo_dir, "images/statistics/")
    os.makedirs(base_path, exist_ok=True)
    for data_loader, loader_type in zip(
        [train_dataloader, valid_dataloader], ["Training", "Test"]
    ):
        num_atoms_list = []
        num_waters_list = []
        for ind, batch in enumerate(data_loader):
            num_atoms = torch.bincount(batch["pro"].batch).numpy()
            num_waters = torch.bincount(batch["wat"].batch).numpy()
            num_atoms_list.extend(num_atoms)
            num_waters_list.extend(num_waters)
            enthalpy.extend(batch["wat"]["enthalpy"])
            entropy.extend(batch["wat"]["entropy"])
        # create histogram of number of atoms and waters
        save_path = os.path.join(base_path, f"histogram_atoms_{loader_type}.svg")
        label = "Number of Atoms"
        plot_frequency(
            num_atoms_list,
            label,
            f"{loader_type} set: Histogram of number of atoms per protein",
            save_path,
            color="red",
        )
        ###############
        save_path = os.path.join(base_path, f"histogram_waters_{loader_type}.svg")
        label = "Number of waters"
        plot_frequency(
            num_waters_list,
            label,
            f"{loader_type} set: Histogram of number of waters per protein",
            save_path,
            color="blue",
        )
        ###############
        enthalpy_tensor = torch.tensor(enthalpy)
        save_path = os.path.join(base_path, f"histogram_enthalpy_{loader_type}.svg")
        label = "enthalpy of waters"
        plot_frequency(
            enthalpy_tensor,
            label,
            f"{loader_type} set: Histogram of enthalpy",
            save_path,
            color="blue",
            bins=30,
        )
        ###############
        entropy_tensor = torch.tensor(entropy)
        save_path = os.path.join(base_path, f"histogram_entropy_{loader_type}.svg")
        label = "entropy of waters"
        plot_frequency(
            entropy_tensor,
            label,
            f"{loader_type} set: histogram of entropy",
            save_path,
            color="blue",
            bins=30,
        )


@hydra.main(
    config_path="../../../config/", config_name="location_model", version_base="1.1"
)
def main(config):
    num_water_statistics(config)


if __name__ == "__main__":
    main()
