"""Evaluate the model after clustering
"""

import resource

import hydra
import torch

from hs_prediction.data.dataset import create_dataloaders, create_inference_dataloader
from hs_prediction.models.location.model import create_model
from hs_prediction.utils.auxiliary import filter_predictions, set_seed
from hs_prediction.utils.clustering import cluster_predictions
from inference.evaluation.evaluate import process_sample


@hydra.main(
    config_path="../../config", config_name="location_model", version_base="1.1"
)
def main(config):
    device = config.inference.cuda_ids[0]
    # Create train and validation dataloaders
    train_dataloader, valid_dataloader, _, _ = create_dataloaders(config)
    # valid_dataloader = create_inference_dataloader(config)
    data_loader = valid_dataloader
    #data_loader = train_dataloader
    model = create_model(config)
    # Load checkpoint
    model.load_checkpoint(config.training.resume_path)
    model = model.to(device)
    num_true_hits = []
    num_pred_hits = []
    num_true_hits_per_occupancy = []
    num_filtered_waters_per_occupancy = []
    num_true_waters = []
    num_predictions = []
    radius_cutoffs = [0.5, 1.0,1.5,2.0]
    occupancy_min = 0.5
    occupancy_max = 1.0
    water_distance_cutoff_min = 0.0  # 0
    water_distance_cutoff_max = 100  # 3.5
    num_points = int((occupancy_max - occupancy_min) / 0.1) + 1
    occupancy_range = torch.linspace(occupancy_min, occupancy_max, num_points)
    for batch_nr, batch in enumerate(data_loader):
        # if batch_nr > 100:
        #    break
        batch = batch.to(device)
        print(batch_nr)
        with torch.no_grad():
            certainty, water_prediction_pos_batch, water_batch_info = model.predict(
                batch
            )
        certainty_cutoff = config.evaluation.certainty_cutoff
        cluster_centers, cluster_probs, cluster_certainty = cluster_predictions(
            water_prediction_pos_batch,
            certainty,
            certainty_cutoff,
            # dst_threshold=1.8,
            dst_threshold=config.evaluation.dst_threshold,
            cluster_aggregation_type="merge",
        )

        certainties_filtered, centers_filtered = filter_predictions(
            cluster_centers,
            cluster_certainty,
            config.evaluation.cluster_certainty_cutoff,
        )
        (
            num_true_hits_batch,
            num_pred_hits_batch,
            num_true_hits_per_occupency_batch,
            num_filtered_waters_per_occupency_batch,
            num_true_waters_batch,
            num_predictions_batch,
        ) = process_sample(
            batch,
            radius_cutoffs,
            occupancy_range,
            torch.tensor(centers_filtered, device=device),
            water_distance_cutoff_min=water_distance_cutoff_min,
            water_distance_cutoff_max=water_distance_cutoff_max,
        )
        num_true_hits.extend(num_true_hits_batch)
        num_pred_hits.extend(num_pred_hits_batch)
        num_true_hits_per_occupancy.extend(num_true_hits_per_occupency_batch)
        num_filtered_waters_per_occupancy.extend(
            num_filtered_waters_per_occupency_batch
        )
        num_true_waters.extend(num_true_waters_batch)
        num_predictions.extend(num_predictions_batch)

    num_true_waters_tensor = torch.tensor(num_true_waters)
    num_true_hits_tensor = torch.tensor(num_true_hits)  # number of truth found
    num_pred_hits_tensor = torch.tensor(
        num_pred_hits
    )  # number of predictions that make sense
    num_true_hits_per_occupancy_tensor = torch.tensor(num_true_hits_per_occupancy)
    num_filtered_waters_per_occupancy_tensor = torch.tensor(
        num_filtered_waters_per_occupancy
    )
    num_predictions_tensor = torch.tensor(num_predictions)
    total_num_waters = num_true_waters_tensor.sum()
    total_trues_found = num_true_hits_tensor.sum(dim=0)
    total_hits = num_pred_hits_tensor.sum(dim=0)
    total_trues_found_per_occupancy = num_true_hits_per_occupancy_tensor.sum(dim=0)
    total_num_waters_per_occupancy = num_filtered_waters_per_occupancy_tensor.sum(dim=0)
    total_num_predictions = num_predictions_tensor.sum(dim=0)
    for ind, cutoff in enumerate(radius_cutoffs):
        print(f"Cutoff {cutoff}: ")
        print(
            "ground truth recovery rate",
            (total_trues_found[ind] / total_num_waters).item(),
        )
        print(
            "prediction hit rate",
            (total_hits[ind] / total_num_predictions).item(),
        )
        for ind_occ, limit_lower, limit_upper in zip(
            range(len(occupancy_range) - 1), occupancy_range[:-1], occupancy_range[1:]
        ):
            print(
                f"Occupancy Interval [{limit_lower:.1f},{limit_upper:.1f}]: ",
                "ground truth recovery rate",
                (
                    total_trues_found_per_occupancy[ind_occ, ind]
                    / total_num_waters_per_occupancy[ind_occ]
                ).item(),
            )


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    # torch.use_deterministic_algorithms(True)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
