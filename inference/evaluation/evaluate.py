"""Evaluate the model performance 
"""

import torch


def metrics_calculation(predicted_points, target_points, cutoffs):
    """calculate evaluation metrics for predicted and target points

    Args:
        predicted_points: predicted points
        target_points: target points
        cutoff: the cutoff parameter for the metrics
    Returns:
        num_true_hits: number of true hits
        num_pred_hits: number of predicted hits
    """
    diff_matrix = predicted_points[None, :, :] - target_points[:, None, :]
    dist_matrix = torch.linalg.norm(diff_matrix, dim=-1)
    num_true_hits_list = []
    num_pred_hits_list = []
    for cutoff in cutoffs:
        nearby = dist_matrix < cutoff
        num_true_hits = nearby.any(dim=1).sum().item()
        num_pred_hits = nearby.any(dim=0).sum().item()
        num_true_hits_list.append(num_true_hits)
        num_pred_hits_list.append(num_pred_hits)
    return num_true_hits_list, num_pred_hits_list


def filter_by_occupancy(water_prediction_pos, occupancy, limit_lower, limit_upper):
    """Filter water prediction positions by occupancy limits.

    Args:
        water_prediction_pos: Tensor of water prediction positions.
        occupancy: Tensor of occupancy values.
        limit_lower: Lower limit for occupancy.
        limit_upper: Upper limit for occupancy.

    Returns:
        Filtered tensor of water prediction positions.
    """
    occupancy_mask = (limit_upper >= occupancy) & (occupancy >= limit_lower)
    return water_prediction_pos[occupancy_mask]


def process_sample(
    batch,
    cutoffs,
    occupancy_range,
    water_predictions_batch,
    water_distance_cutoff_min=None,
    water_distance_cutoff_max=None,
    water_batch_info=None,
):
    """Calculate the prediction hit rate and ground truth recovery rate. Also provide results for different occupancy ranges.

    Args:
        batch: The data batch containing protein and water information.
        cutoffs: distance cutoff for the metrics (what it means that a prediction is correct)
        occupancy_range: anaylize the metrics for waters within the occupancy ranges
        water_predictions_batch: which predictions are belonging to which sample in batch
        water_distance_cutoff_min: only consider true waters having at least this distance to the protein
        water_distance_cutoff_max: only consider true waters at most this distance to the protein
        water_batch_info: _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if water_batch_info is None:
        water_batch_info = torch.zeros(water_predictions_batch.shape[0])
    num_true_hits = []
    num_pred_hits = []
    num_true_hits_per_occupancy = []
    num_filtered_waters_per_occupancy = []
    num_true_waters = []
    num_predictions = []
    batch_elements = torch.unique(water_batch_info)
    true_water_pos_batch = batch["wat"].pos
    true_water_batch = batch["wat"].batch
    for batch_element in batch_elements:
        water_prediction_pos = water_predictions_batch[
            batch_element == water_batch_info
        ]
        occupancies = batch["wat"].occupancy[batch_element == batch["wat"].batch]
        true_water_pos = true_water_pos_batch[batch_element == true_water_batch]
        if (
            water_distance_cutoff_min is not None
            or water_distance_cutoff_max is not None
        ):
            differences = true_water_pos[:, None] - batch["pro"]["pos"][None]
            distances = torch.norm(differences, dim=-1)
            distance_mask = torch.ones(true_water_pos.shape[0], dtype=torch.bool)
            if water_distance_cutoff_min is not None:
                distance_min_mask = torch.any(
                    distances < water_distance_cutoff_min, dim=-1
                )
                distance_mask[distance_min_mask] = False
            if water_distance_cutoff_max is not None:
                distance_max_mask = torch.all(
                    distances > water_distance_cutoff_max, dim=-1
                )
                distance_mask[distance_max_mask] = False
            true_water_pos = true_water_pos[distance_mask]
            occupancies = occupancies[distance_mask]

        num_true_hits_per_occupancy_sample = []
        num_filtered_waters_per_occupancy_sample = []

        for limit_lower, limit_upper in zip(occupancy_range[:-1], occupancy_range[1:]):
            filtered_true_water_pos = filter_by_occupancy(
                true_water_pos, occupancies, limit_lower, limit_upper
            )
            num_true_hits_list, _ = metrics_calculation(
                water_prediction_pos, filtered_true_water_pos, cutoffs
            )
            num_true_hits_per_occupancy_sample.append(num_true_hits_list)
            num_filtered_waters_per_occupancy_sample.append(
                filtered_true_water_pos.shape[0]
            )
        num_true_hits_per_occupancy.append(num_true_hits_per_occupancy_sample)
        num_filtered_waters_per_occupancy.append(
            num_filtered_waters_per_occupancy_sample
        )
        num_true_waters.append(true_water_pos.shape[0])
        num_true_hits_list_full, num_pred_hits_list = metrics_calculation(
            water_prediction_pos, true_water_pos, cutoffs
        )
        num_true_hits.append(num_true_hits_list_full)
        num_pred_hits.append(num_pred_hits_list)
        num_predictions.append(water_prediction_pos.shape[0])
        for ind, cutoff in enumerate(cutoffs):
            if num_true_waters[0] > 0 and num_predictions[0] > 0:
                print(f"Cutoff {cutoff}: ")
                print(
                    "ground truth recovery rate",
                    num_true_hits_list_full[ind] / num_true_waters[0],
                )
                print(
                    "prediction hit rate",
                    num_pred_hits_list[ind] / num_predictions[0],
                )
    return (
        num_true_hits,
        num_pred_hits,
        num_true_hits_per_occupancy,
        num_filtered_waters_per_occupancy,
        num_true_waters,
        num_predictions,
    )
