"""Clustering predictions
"""
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from hs_prediction.utils.auxiliary import filter_predictions


def merge_clusters(cluster_classes, predicted_positions, predicted_probs):
    """Calculate (weighted) cluster centers and cluster probabilities

    Args
        cluster_classes: determines to which cluster point belongs
        predicted_positions: predicted field points
        predicted_probs: predicted probabilities of field points

    Return:
        cluster_centers: centers of clusters
        cluster_center_prob: sums of probabilities of points contained in
             corresponding clusters
    """
    clusters = np.unique(cluster_classes)
    cluster_centers = []
    cluster_center_prob = []
    for _, cluster in enumerate(clusters):
        cluster_idx = cluster_classes == cluster
        cluster_points = predicted_positions[cluster_idx, :]
        cluster_probs = predicted_probs[cluster_idx]
        cluster_proportions = cluster_probs / np.sum(cluster_probs)
        cluster_centers.append(
            np.sum(cluster_proportions.reshape((-1, 1)) * cluster_points, 0)
        )
        cluster_center_prob.append(np.sum(cluster_probs))
    return cluster_centers, cluster_center_prob


def arg_max_clusters(cluster_classes, predicted_positions, predicted_probs):
    """Calculate cluster centers by taking the point with the highest probability

    Args:
        cluster_classes: determines to which cluster point belongs
        predicted_positions: predicted field points
        predicted_probs: predicted probabilities of field points

    Return:
        cluster_centers: centers of clusters based on max probability
        cluster_center_prob: max probabilities of points contained in corresponding clusters
    """
    clusters = np.unique(cluster_classes)
    cluster_centers = []
    cluster_center_prob = []
    for _, cluster in enumerate(clusters):
        cluster_idx = cluster_classes == cluster
        cluster_points = predicted_positions[cluster_idx, :]
        cluster_probs = predicted_probs[cluster_idx]
        # Find the index of the maximum probability in the cluster
        max_prob_idx = np.argmax(cluster_probs)
        # Use the point with the maximum probability as the cluster center
        cluster_centers.append(cluster_points[max_prob_idx])
        cluster_center_prob.append(np.sum(cluster_probs))

    return cluster_centers, cluster_center_prob


def cluster_preds(predicted_fp, predicted_probs, cluster_model, filter_level=0.005):
    """Calculate (weighted) cluster centers and cluster probabilities

    Args:
        predicted_fp: predicted positions of field points
        predicted_probs:predicted probability weights for fieldpoints
        cluster_model: the cluster model to assign clusters to points
        filter_level: the threshold for probability to make a prediction

    Return:
        cluster_centers_valid: centers of clusters which have high enough probability
        cluster_center_probs_valid: probability sum of of points in cluster, for clusters
            which exceed 'filter_level' probability

    """
    # only keep larger probs:
    valid_preds = predicted_probs > 0.0001
    predicted_probs_valid = predicted_probs[valid_preds]
    predicted_fp_valid = predicted_fp[valid_preds]
    # AgglomerativeClustering only works for at least 2 points:
    if torch.sum(valid_preds) == 1:
        cluster_centers_valid = predicted_fp_valid
        cluster_probs_valid = predicted_probs_valid
        return cluster_centers_valid, cluster_probs_valid
    cluster_classes = cluster_model.fit_predict(predicted_fp_valid.cpu())
    cluster_centers, cluster_probs = merge_clusters(
        cluster_classes,
        np.array(predicted_fp_valid.cpu()),
        np.array(predicted_probs_valid.cpu()),
    )
    cluster_centers = np.array(cluster_centers)
    cluster_probs = np.array(cluster_probs)
    # only make predictions at points with at least 'filter_level' probability
    valid_clusters = cluster_probs > filter_level
    cluster_centers_valid = cluster_centers[valid_clusters]
    cluster_probs_valid = cluster_probs[valid_clusters]
    return cluster_centers_valid, cluster_probs_valid


def cluster_predictions(
    predicted_points,
    certainties,
    certainty_cutoff,
    dst_threshold=0.5,
    cluster_aggregation_type="merge",
):
    """cluter predictions based on distance threshold"""
    certainties_filtered, predicted_points_filtered = filter_predictions(
        predicted_points, certainties, certainty_cutoff
    )
    predicted_probs = certainties_filtered / torch.sum(certainties_filtered)
    cluster_model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dst_threshold
    )
    cluster_classes = cluster_model.fit_predict(predicted_points_filtered.cpu())

    if cluster_aggregation_type == "merge":
        cluster_centers, cluster_probs = merge_clusters(
            cluster_classes,
            np.array(predicted_points_filtered.cpu()),
            np.array(predicted_probs.cpu()),
        )
    elif cluster_aggregation_type == "max":
        # Implement the max aggregation logic here
        cluster_centers, cluster_probs = arg_max_clusters(
            cluster_classes,
            np.array(predicted_points_filtered.cpu()),
            np.array(predicted_probs.cpu()),
        )
    else:
        raise ValueError(
            f"Unsupported cluster_aggregation_type: {cluster_aggregation_type}"
        )

    cluster_centers = np.array(cluster_centers)
    cluster_probs = np.array(cluster_probs)
    cluster_centertainties = cluster_probs * np.array(
        torch.sum(certainties_filtered).cpu()
    )
    cluster_centertainties = np.array(cluster_centertainties)
    return cluster_centers, cluster_probs, cluster_centertainties


def predict_locations(
    location_model, batch, certainty_cutoff, dst_threshold, cluster_certainty_cutoff
):
    with torch.no_grad():
        certainty, water_prediction_pos_batch, water_batch_info = (
            location_model.predict(batch)
        )
    cluster_centers, cluster_probs, cluster_certainty = cluster_predictions(
        water_prediction_pos_batch,
        certainty,
        certainty_cutoff,
        dst_threshold=dst_threshold,
        cluster_aggregation_type="merge",
    )
    certainties_filtered, centers_filtered = filter_predictions(
        cluster_centers, cluster_certainty, cluster_certainty_cutoff
    )
    return certainties_filtered, centers_filtered
