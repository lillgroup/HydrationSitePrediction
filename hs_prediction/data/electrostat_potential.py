"""Calculate the electrostatic potential of water molecules with respect to the protein. Note that this feature is not used in the current version of the model."""

import numpy as np
import pandas as pd
import parmed
import os
import hydra


def calculate_coloumb_potential(
    path_to_water, path_charge_info, protein_path, output_path
):
    """Calculate the electrostatic potential of water molecules with respect to the protein"""
    watsite_df = pd.read_csv(path_to_water)
    wat_coords = watsite_df[["center_x", "center_y", "center_z"]].values.astype(
        np.float16
    )

    topology = parmed.load_file(path_charge_info)
    charge_data = {}
    for atom in topology.atoms:
        charge_data[f"{atom.name}-{atom.residue.name}-{atom.residue.idx+1}"] = (
            atom.charge
        )
    pro_coords = []
    pro_charges = []
    for l in open(protein_path, "r").read().split("\n"):
        if l.startswith("ATOM"):
            pro_coords.append([l[30:38], l[38:46], l[46:54]])
            key = f"{l[12:16].strip()}-{l[17:20].strip()}-{l[22:26].strip()}"
            if key not in charge_data:
                print(key)
            charge = charge_data[key] if key in charge_data else 0.0
            pro_charges.append(charge)
    pro_coords = np.array(pro_coords, dtype=float)
    pro_charges = np.array(pro_charges, dtype=float)
    wat_pro_dis = np.linalg.norm(
        wat_coords.reshape(-1, 1, 3) - pro_coords.reshape(1, -1, 3), axis=-1
    ).astype(np.float16)
    qqr = np.tile(-0.82 * pro_charges, (len(wat_coords), 1)) / wat_pro_dis
    mask = wat_pro_dis < 10.0
    electrostatic_potential = np.sum(qqr * mask, axis=1)
    np.save(output_path, electrostatic_potential.astype(np.float16))


@hydra.main(
    config_path="../../config", config_name="location_model", version_base="1.1"
)
def main(config):
    if config.data.name != "watsite":
        raise ValueError("Please change the data set used to 'watsite'")
    watsite_path = os.path.dirname(config.data.dataset_path)
    pdb_id = "1m21"
    csv_path = os.path.join(watsite_path, "data", pdb_id, "watsite.csv")
    top_path = os.path.join(watsite_path, "prmtop_dir", pdb_id, "prepared.prmtop")
    pro_path = os.path.join(watsite_path, "data", pdb_id, "protein.pdb")
    out_path = os.path.join(watsite_path, "data", pdb_id, "energies.npy")
    calculate_coloumb_potential(csv_path, top_path, pro_path, out_path)


if __name__ == "__main__":
    #example usage:
    main()
