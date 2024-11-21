"""For the displaced waters, we calculate the entropy/enthalpy of the displaced waters
"""

import glob
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.core.global_hydra import GlobalHydra
from scipy.stats import pearsonr

from hs_prediction.data.pdb import PDB
from inference.evaluation.predict_waters import create_water_predictions


def create_scatter_plot(
    experimental: torch.Tensor,
    predicted: torch.Tensor,
    title: str,
    output_file: str,
):
    experimental_np = experimental.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    coefficients = np.polyfit(experimental_np, predicted_np, 1)
    linear_fit = np.poly1d(coefficients)
    x_values = np.linspace(
        min(experimental_np) - 1, max(experimental_np) + 1, 100
    )  # Create evenly spaced x-values
    y_values = linear_fit(x_values)
    plt.plot(x_values, y_values, color="grey", linestyle="--")
    plt.scatter(experimental_np, predicted_np, alpha=1.0)
    plt.xlabel("Experimental")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.savefig(output_file)
    plt.close()


def displaced_waters_indices(waters, ligand_atoms, cutoff):
    """Calculate which waters are displaced by the ligand

    Args:
        waters: coordinates of the predicted waters
        ligand_atoms: coordinates of the predicted ligand atoms
        cutoff: if a ligand atom is within cutoff distance of a water atom, the water is considered displaced
    """
    differences = waters[:, None] - ligand_atoms[None]
    distances = torch.norm(differences, dim=-1)
    closer_than_cutoff = torch.any(distances < cutoff, dim=1)
    water_indices = torch.nonzero(closer_than_cutoff, as_tuple=True)[0]
    return water_indices


def thermodynamics_displaced_waters(
    waters, enthalpies, entropies, ligand_atoms, cutoff
):
    water_indices = displaced_waters_indices(waters, ligand_atoms, cutoff)
    entropies_sum = entropies[water_indices].sum()
    enthalpies_sum = enthalpies[water_indices].sum()
    return entropies_sum, enthalpies_sum


def calculate_binding_free_energy(base_dir_predictions, displacement_cutoff, base_dir):
    binding_free_energies = {}
    for dirpath, dirnames, filenames in os.walk(base_dir_predictions):
        # Skip the base directory itself
        if dirpath == base_dir_predictions:
            continue
        path_water = os.path.join(dirpath, "location_prediction.pt")
        path_enthalpy = os.path.join(dirpath, "enthalpy.pt")
        path_entropy = os.path.join(dirpath, "entropy.pt")
        enthalpy = torch.load(path_enthalpy, weights_only=True)
        entropy = torch.load(path_entropy, weights_only=True)
        waters = torch.tensor(torch.load(path_water, weights_only=False))
        protein_name = os.path.basename(dirpath)
        binding_free_energies[protein_name] = {}
        ligand_files = glob.glob(os.path.join(base_dir, protein_name, "ligand*.pdb"))
        for file_path in ligand_files:
            ligand = PDB.from_file(file_path)
            ligand_coords = torch.tensor(ligand.get_coords())
            entropies_sum, enthalpies_sum = thermodynamics_displaced_waters(
                waters, enthalpy, entropy, ligand_coords, displacement_cutoff
            )
            binding_free_energy = enthalpies_sum - entropies_sum
            ligand_name = os.path.basename(file_path)
            binding_free_energies[protein_name][
                ligand_name
            ] = binding_free_energy.item()
    return binding_free_energies


def main():
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../config/", version_base="1.1")
    config_thermo = hydra.compose(config_name="thermo_model")
    config_location = hydra.compose(config_name="location_model")
    # hrow error if config.data is not set
    if (
        config_thermo.data.name != "case_study"
        or config_location.data.name != "case_study"
    ):
        raise ValueError(
            "Please change the data set used to 'case_study' in the config files 'config/thermo_model.yaml' and 'config/location_model.yaml'"
        )

    base_dir_predictions = os.path.join(
        config_thermo.general.repo_dir, "images/case_study"
    )

    config_location.evaluation.certainty_cutoff = 0.023
    config_location.evaluation.cluster_certainty_cutoff = 0.023
    create_water_predictions(
        base_dir_predictions, config_thermo, config_location, False
    )
    for displacement_cutoff in torch.arange(2.0, 2.6, 0.1):
        desolvation_free_energies = calculate_binding_free_energy(
            base_dir_predictions, displacement_cutoff, config_thermo.data.dataset_path
        )
        # read affinity
        path_experimental_affinity = os.path.join(
            os.path.dirname(config_thermo.data.dataset_path), "affinity/affinity.csv"
        )
        df = pd.read_csv(path_experimental_affinity)
        free_energies_reordered = []
        for ind, protein in enumerate(df["pdb"]):
            ligand_keys = desolvation_free_energies[protein].keys()
            if len(ligand_keys) == 1:
                free_energies_reordered.append(
                    desolvation_free_energies[protein][list(ligand_keys)[0]]
                )
            else:
                ligand_name = df["lig"][ind]
                if ligand_name == "ET":
                    # ligand_name = "ET1"
                    ligand_name = "ET2"
                free_energies_reordered.append(
                    desolvation_free_energies[protein][f"ligand_{ligand_name}.pdb"]
                )
        plot_path = os.path.join(
            base_dir_predictions,
            f"prediction_vs_experiment_correlation_displacement_{displacement_cutoff}A.svg",
        )
        pearson_corr_free_energy, _ = pearsonr(
            torch.tensor(list(df["exp"])).cpu().numpy(),
            torch.tensor(free_energies_reordered).cpu().numpy(),
        )
        print(
            f"pearson r correlation for displacement tolerance {displacement_cutoff}: ",
            pearson_corr_free_energy,
        )
        create_scatter_plot(
            torch.tensor(list(df["exp"])),
            torch.tensor(free_energies_reordered),
            "Desolvation free energies vs experimental binding affinities",
            plot_path,
        )


if __name__ == "__main__":
    main()
