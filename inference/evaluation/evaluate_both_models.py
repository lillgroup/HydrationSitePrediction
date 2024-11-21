""" Given a protein structure, predict both the location of hydration sites and the thermodynamic properties of the water molecules.
"""

import resource

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from scipy.spatial import KDTree

from hs_prediction.data.dataset import create_inference_dataloader
from hs_prediction.models.location.model import create_model as create_location_model
from hs_prediction.models.thermodynamics.model_entropy import create_thermo_model
from hs_prediction.utils.auxiliary import set_seed
from hs_prediction.utils.clustering import predict_locations


def predict_thermodynamics(model_thermo, batch):
    with torch.no_grad():
        thermodynamics_predicted = model_thermo.model(batch)
    predicted_enthalpy = thermodynamics_predicted[:, 0]
    predicted_entropy = thermodynamics_predicted[:, 1]
    true_enthalpy = batch["wat"].enthalpy[:]
    true_entropy = batch["wat"].entropy[:]
    return predicted_enthalpy, predicted_entropy, true_enthalpy, true_entropy


def find_nearest_neighbor(predicted_waters, true_waters):
    """Find nearest neighbor of predicted waters in true waters

    Args:
        predicted_waters: predicted waters (n x 3 array)
        true_waters: ground truth waters (m x 3 array)
    """
    tree = KDTree(true_waters)
    distances, indices = tree.query(predicted_waters)
    return distances, indices


def create_model_location(config):
    model = create_location_model(config)
    model.load_checkpoint(config.training.resume_path)
    device = config.inference.cuda_ids[0]
    model = model.to(device)
    return model


def create_model_thermo(config):
    model = create_thermo_model(config)
    model.load_checkpoint(config.training.resume_path)
    device = config.inference.cuda_ids[0]
    model = model.to(device)
    return model


def predict_water_with_thermos(model_location, config_location, model_thermo, batch):
    """Use both models to predict locations and thermodynamic properties of waters

    Args:
        model_location: model for location prediction
        config_location: configurations for location model
        model_thermo: model for predicting entropy and enthalpy
        batch: a protein

    Returns:
        _type_: _description_
    """
    device = config_location.inference.cuda_ids[0]
    certainty_prediction, location_prediction = predict_locations(
        model_location,
        batch,
        config_location.evaluation.certainty_cutoff,
        config_location.evaluation.dst_threshold,
        config_location.evaluation.cluster_certainty_cutoff,
    )
    batch_prediction = {
        "wat": {
            "pos": torch.tensor(location_prediction, device=device),
            "batch": torch.zeros(
                location_prediction.shape[0], dtype=torch.long, device=device
            ),
        },
        "pro": {
            "features": batch["pro"].features.clone().detach().to(device),
            "pos": batch["pro"].pos.clone().detach().to(device),
            "put_wat": batch["pro"].put_wat.clone().detach().to(device),
            "batch": batch["pro"].batch.clone().detach().to(device),
            "ptr": batch["pro"].ptr.clone().detach().to(device),
        },
    }
    with torch.no_grad():
        thermodynamics_predicted = model_thermo.model(batch_prediction)
        enthalpy = thermodynamics_predicted[:, 0]
        entropy = thermodynamics_predicted[:, 1]
    return location_prediction, enthalpy, entropy


def main():
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../config/", version_base="1.1")
    # config_thermo = hydra.compose(config_name="gat1")
    config_thermo = hydra.compose(config_name="thermo_model")
    # config_thermo = hydra.compose(config_name="gat2")
    config_location = hydra.compose(config_name="location_model")
    valid_dataloader = create_inference_dataloader(config_location)
    device = config_location.inference.cuda_ids[0]
    model_location = create_model_location(config_location)
    model_thermo = create_model_thermo(config_thermo)
    config_location.data.thermodynamics = True
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    for near_cutoff in cutoffs:
        data_loader = valid_dataloader
        enthalpies_predicted_list = []
        entropies_predicted_list = []
        enthalpies_nearby_list = []
        entropies_nearby_list = []
        for batch_nr, batch in enumerate(data_loader):
            if batch_nr > 100:
                break
            batch = batch.to(device)
            location_prediction, enthalpy, entropy = predict_water_with_thermos(
                model_location, config_location, model_thermo, batch
            )
            distances, indices = find_nearest_neighbor(
                location_prediction, batch["wat"]["pos"].cpu()
            )
            near_mask = distances < near_cutoff
            enthalpy_selected = enthalpy[near_mask]
            entropy_selected = entropy[near_mask]
            inidices_selected = indices[near_mask]
            enthalpy_nearby = batch["wat"].enthalpy[inidices_selected]
            entropy_nearby = batch["wat"].entropy[inidices_selected]
            mse_enthalpy = torch.mean((enthalpy_selected - enthalpy_nearby) ** 2)
            mse_entropy = torch.mean((entropy_selected - entropy_nearby) ** 2)
            enthalpies_predicted_list.append(enthalpy_selected)
            enthalpies_nearby_list.append(enthalpy_nearby)
            entropies_predicted_list.append(entropy_selected)
            entropies_nearby_list.append(entropy_nearby)
        enthalpies_predicted_total = torch.cat(enthalpies_predicted_list)
        entropies_predicted_total = torch.cat(entropies_predicted_list)
        enthalpies_nearby_total = torch.cat(enthalpies_nearby_list)
        entropies_nearby_total = torch.cat(entropies_nearby_list)
        mse_enthalpy = torch.mean(
            (enthalpies_predicted_total - enthalpies_nearby_total) ** 2
        )
        mse_entropy = torch.mean(
            (entropies_predicted_total - entropies_nearby_total) ** 2
        )
        print(f"near cutoff: {near_cutoff}")
        print("enthalpy mse:", mse_enthalpy)
        print("entropy mse:", mse_entropy)


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    # torch.use_deterministic_algorithms(True)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
