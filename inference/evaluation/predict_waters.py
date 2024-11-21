"""predict waters and thermodynamic properties of waters
"""

import os
import resource

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra

from hs_prediction.data.dataset import create_inference_dataloader
from hs_prediction.utils.auxiliary import set_seed
from inference.evaluation.evaluate_both_models import (
    create_model_location,
    create_model_thermo,
    predict_water_with_thermos,
)
from inference.visualization.visualize import create_pymol_protein_with_predictions


def create_water_predictions(base_path, config_thermo, config_location, visualize=True):
    """predict water locations and thermodynamic properties of waters and visualize it

    Args:
        save_path: path to dir where to save the predictions
    """
    device = config_location.inference.cuda_ids[0]
    valid_dataloader = create_inference_dataloader(config_location)
    model_location = create_model_location(config_location)
    model_thermo = create_model_thermo(config_thermo)
    data_loader = valid_dataloader
    for batch_nr, batch in enumerate(data_loader):
        batch = batch.to(device)
        print(batch_nr)
        location_prediction, enthalpy, entropy = predict_water_with_thermos(
            model_location, config_location, model_thermo, batch
        )
        path = os.path.join(base_path, batch["key"][0])
        # save to file
        os.makedirs(path, exist_ok=True)
        torch.save(location_prediction, path + "/location_prediction.pt")
        torch.save(enthalpy, path + "/enthalpy.pt")
        torch.save(entropy, path + "/entropy.pt")
        true_waters_path = os.path.join(
            config_location.data.dataset_path,
            f"{batch['key'][0]}/water_coordinates.npy",
        )
        if os.path.exists(true_waters_path):
            true_waters = torch.tensor(np.load(true_waters_path))
            torch.save(true_waters, path + "/true_waters.pt")
        else:
            true_waters = None
        if visualize:
            protein_pdb_path = (
                config_location.data.dataset_path
                + f"/{batch['key'][0]}/protein_sasa.pdb"
            )
            output_path = f"{path}/protein_with_predictions.pse"
            create_pymol_protein_with_predictions(
                protein_pdb_path,
                location_prediction,
                enthalpy,
                entropy,
                true_waters,
                output_path,
            )


def main():
    # base_path = "./images/protein_examples/"
    base_path = "./images/case_study/"
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../config/", version_base="1.1")
    config_thermo = hydra.compose(config_name="thermo_model")
    config_location = hydra.compose(config_name="location_model")
    create_water_predictions(base_path, config_thermo, config_location)


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    # torch.use_deterministic_algorithms(True)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
