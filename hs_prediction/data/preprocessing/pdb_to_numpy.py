"""Read all pdb files and save numpy arrays
"""

import os
import numpy as np
from Bio.PDB import PDBParser
import pandas as pd
import hydra


def read_water_pdb(file_path):
    """Read coordinates from a water.pdb file using Biopython."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("water", file_path)
    coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == "O":  # Only consider oxygen atoms
                        coord = atom.coord
                        coordinates.append(coord)

    return np.array(coordinates)


def process_directories(root_dir, water_file_name="water.pdb"):
    """Traverse directories, read PDB files, and save coordinates as NumPy arrays and CSV files."""
    processed_dirs = []
    for subdir, _, files in os.walk(root_dir):
        if water_file_name in files:
            dir_name = os.path.basename(subdir)
            processed_dirs.append(dir_name)
            pdb_path = os.path.join(subdir, water_file_name)
            coords = read_water_pdb(pdb_path)
            output_npy_path = os.path.join(subdir, "water_coordinates.npy")
            output_csv_path = os.path.join(subdir, "water_coordinates.csv")

            # Save as .npy
            np.save(output_npy_path, coords)
            print(f"Saved coordinates for {pdb_path} to {output_npy_path}")

            # Save as .csv
            df = pd.DataFrame(coords, columns=["center_x", "center_y", "center_z"])
            df.to_csv(output_csv_path, index=False)
            print(f"Saved coordinates for {pdb_path} to {output_csv_path}")

    return processed_dirs


def log_directories(directories, log_file_path):
    """Log the directory names to a text file."""
    with open(log_file_path, "w") as log_file:
        for directory in directories:
            log_file.write(directory + "\n")


@hydra.main(
    config_path="../../../config", config_name="location_model", version_base="1.1"
)
def main(config):
    if config.data.name not in ["protein_examples", "protein_examples2"]:
        raise ValueError(
            "Please change the data set used to 'protein_examples' or 'protein_examples2' in the config file"
        )
    dataset_path = config.data.dataset_path
    water_file_name = "waters.pdb"
    processed_dirs = process_directories(dataset_path, water_file_name)
    log_file_path = os.path.join(os.path.dirname(dataset_path), "test.txt")
    log_directories(processed_dirs, log_file_path)


if __name__ == "__main__":
    main()
