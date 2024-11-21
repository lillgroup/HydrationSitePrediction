"""Create visualizations of the predictions after initial placement and each layer of the model
"""

import os.path
import resource
from os.path import join

import hydra
import torch

from hs_prediction.data.dataset import create_inference_dataloader
from hs_prediction.models.location.model import create_model
from hs_prediction.utils.auxiliary import filter_certainty, filter_predictions
from hs_prediction.utils.clustering import cluster_predictions
from inference.visualization.visualize import (
    create_pymol_objects_in_states,
    plot_3d_points_interactive,
)


def visualize_prediction_layers(protein_pdb_path, true_coords, sample_dir):
    protein_pdb_path = torch.load(
        os.path.join(sample_dir, "pdb_path.pt"), weights_only=False
    )
    files = [
        "placement.pt",
        "water_pred_layer_0.pt",
        "water_pred_layer_1.pt",
        "water_pred_layer_2.pt",
        "water_pred_layer_3.pt",
        "water_pred_layer_4.pt",
        "water_pred_layer_5.pt",
        "after_clustering.pt",
    ]
    file_paths = [os.path.join(sample_dir, file) for file in files]
    predictions = []
    for file_path in file_paths:
        predictions.append(torch.load(file_path, weights_only=False))
    true_coord_path = os.path.join(sample_dir, "true_water.pt")
    true_coords = torch.load(true_coord_path, weights_only=True)
    true_coords_list = [true_coords] * len(predictions)
    create_pymol_objects_in_states(
        protein_pdb_path, predictions, true_coords_list, sample_dir
    )


@hydra.main(
    config_path="../../config/", config_name="location_model", version_base="1.1"
)
def main(config):
    if config.data.name != "watsite":
        raise ValueError(
            "Please change the data set used to 'watsite' in the config file"
        )
    repo_dir = config.general.repo_dir
    device = config.inference.cuda_ids[0]
    test_dataloader = create_inference_dataloader(config)
    model = create_model(config)
    # Load checkpoint
    load_model = True
    if load_model:
        base_dir = os.path.join(repo_dir, "images/trajectory_after_training/")
        model.load_checkpoint(config.training.resume_path)
    else:
        base_dir = os.path.join(repo_dir, "images/trajectory_before_training/")
    model = model.to(device)
    for ind, batch in enumerate(test_dataloader):
        if ind > 1:
            break
        batch = batch.to(device)
        key = batch.key[0]
        save_path = join(base_dir, batch["key"][0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path_to_pdb = join(config.data.dataset_path, key, "protein.pdb")
        torch.save(path_to_pdb, join(save_path, "pdb_path.pt"))
        with torch.no_grad():
            certainty, water_prediction_pos, water_batch_info = model.predict(
                batch, save_path
            )
        certainty_mask_batch = filter_certainty(certainty, 0.00)
        # certainty_mask_batch = filter_certainty(certainty, 0.2)
        certainty_mask = certainty_mask_batch.squeeze()
        water_positions = batch["wat"].pos
        occupancy = batch["wat"].occupancy
        sample_dir = os.path.join(base_dir, batch["key"][0])
        output_file = os.path.join(sample_dir, "point_prediction.html")
        predicted_probs = certainty / torch.sum(certainty) * 1000
        water_prediction_pos_filtered = water_prediction_pos[certainty_mask]
        predicted_probs_filtered = predicted_probs[certainty_mask]
        plot_3d_points_interactive(
            water_prediction_pos_filtered.detach().cpu().numpy(),
            predicted_probs_filtered.detach().cpu().numpy(),
            water_positions.detach().cpu().numpy(),
            occupancy.detach().cpu().numpy(),
            output_file,
        )
        cluster_centers, cluster_probs, cluster_certainty = cluster_predictions(
            water_prediction_pos,
            certainty,
            config.evaluation.certainty_cutoff,
            dst_threshold=config.evaluation.dst_threshold,
            cluster_aggregation_type="merge",
        )
        certainties_filtered, centers_filtered = filter_predictions(
            cluster_centers,
            cluster_certainty,
            config.evaluation.cluster_certainty_cutoff,
        )
        torch.save(
            centers_filtered,
            os.path.join(save_path, "after_clustering.pt"),
        )
        visualize_prediction_layers(path_to_pdb, water_positions, sample_dir)


if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
