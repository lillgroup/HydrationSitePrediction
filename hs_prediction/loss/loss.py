"""Loss functions
"""
import torch

def weighted_vector_length_sum(predicted_vectors, predicted_probs):
    """calculate the sum of vector lengths, scaled with corresponding probability

    Args
        predicted_vectors: vectors as predicted by model
        predicted_probs:  probabilities corresponding to vectors

    Return:
        scaled_sum_weighted_vector: rescaled and weighted (by probability) sum of vector lenghts
    """
    vector_lengths = torch.norm(predicted_vectors, dim=2).reshape((-1))
    weighted_vector_lengths = predicted_probs * vector_lengths
    scaled_sum_weighted_vector = 10 * torch.sum(weighted_vector_lengths)
    return scaled_sum_weighted_vector


def kl_approx(f_centers, f_probs, g_centers, g_probs, sigmas):
    """Calculate an approxmation of KL(f|g)

    Args
        f_centers: points of discrete distribution on f_centers
        f_probs: corresponding probabilities for f_centers
        g_centers: centers of gaussian mixture g
        g_weights: corresponding probabilities for g_centers

    Return:
        kl_mc_estimator: approximation of KL(f|g)
    """
    # distence between centers:
    center_diff = torch.unsqueeze(f_centers, dim=1) - torch.unsqueeze(g_centers, dim=0)
    center_dist = torch.norm(center_diff, dim=2)
    sigmas = sigmas.reshape(-1)
    g_val = torch.sum(
        torch.exp(-0.5 * sigmas[:, None] ** (-2) * center_dist**2) * g_probs, dim=1
    )
    # g_val cannot be smaller than 0, it just occurs by numerics
    g_val[g_val < 0] = 0
    eps = 1e-12
    log_f_div_g = torch.log(f_probs / (g_val + eps) + eps)
    kl_mc_estimator = torch.sum(f_probs * log_f_div_g)
    return kl_mc_estimator


def composed_loss(
    true_centers,
    true_weights,
    predicted_points,
    predicted_weights,
    sigmas_true_points,
    sigmas_predicted_points,
    kl_type,
    weights_penalty,
    weights_penalty_scaling,
):
    """Loss function consisting of symmetrized Kullback Leibler divergence, penalization of large
        probability weights, penalization of vector lengths

    Args
        true_centers: location of target points
        true_weights: charges of target points
        predicted_points: location of predicted point
        predicted_weights: predicted probability weights of points
        sigma: standard deviation to use in Gaussian mixture
        predicted_vectors: predicted vectors
        weights_penalty_scaling: scaling parameter for weights penalty

    Return:
        loss_total: sum of different losses (symmetrized Kullback Leibler divergence, penalization of large
        probability weights, penalization of vector lenghts)
    """
    clip_value = 1 / 10000
    clipped_weights = torch.clamp(predicted_weights, min=clip_value)
    predicted_probs = clipped_weights / torch.sum(clipped_weights)
    true_probs = torch.abs(true_weights) / torch.sum(torch.abs(true_weights))
    # kl loss
    kl_approx_result = 0
    reverse_kl_approx_result = 0
    if kl_type == "kl" or kl_type == "kl_sym":
        kl_approx_result = kl_approx(
            true_centers,
            true_probs,
            predicted_points,
            predicted_probs,
            sigmas_true_points,
        )
    elif kl_type == "reverse_kl" or kl_type == "kl_sym":
        reverse_kl_approx_result = kl_approx(
            predicted_points,
            predicted_probs,
            true_centers,
            true_probs,
            sigmas_predicted_points,
        )
    probs_loss = 0
    if weights_penalty:
        probs_loss = -torch.sum(predicted_probs**2)
    scaling_factor = torch.sum(torch.abs(true_weights))
    loss_total = scaling_factor * (
        kl_approx_result
        + reverse_kl_approx_result
        + weights_penalty_scaling * probs_loss
    )
    return loss_total


def loss(
    predicted_certainty, predicted_positions, occupancy, true_positions, loss_config
):
    sigma = loss_config.sigma
    kl_type = loss_config.kl_type
    weights_penalty = loss_config.weights_penalty
    weights_penalty_scaling = loss_config.weights_penalty_scaling
    sigma_min = sigma
    sigma_max = sigma
    sigmas_true_points = sigma_min * occupancy + sigma_max * (1 - occupancy)
    sigmas_predicted_points = torch.ones_like(predicted_certainty.squeeze()) * sigma_min

    loss = composed_loss(
        true_positions,
        occupancy.squeeze(),
        predicted_positions,
        predicted_certainty.squeeze(),
        sigmas_true_points,
        sigmas_predicted_points,
        kl_type,
        weights_penalty,
        weights_penalty_scaling,
    )
    return loss


def batch_loss(
    predicted_certainty,
    predicted_positions,
    batch_info,
    batch_occupancy,
    true_positions,
    true_water_batch,
    loss_function,
    loss_config,
):
    """Iterate over batch elements and apply loss function

    Args:
        predicted_certainty: the certainty of the water prediction (a number between 0 and 1)
        predicted_positions: predicted positions
        batch_info: information about to which sample of the batch a predicted element belongs
        true_positions: true positions of the water molecules
        true_water_batch: batch info of true water molecules
        loss_function: loss function to apply
    Returns:
        average_loss: average loss over all batch elements
    """
    batch_elements = torch.unique(batch_info)
    loss_tot = 0
    for batch_element in batch_elements:
        predicted_certainty_element = predicted_certainty[batch_info == batch_element]
        predicted_positions_element = predicted_positions[batch_info == batch_element]
        true_positions_element = true_positions[true_water_batch == batch_element]
        occupancy = batch_occupancy[true_water_batch == batch_element]
        loss = loss_function(
            predicted_certainty_element,
            predicted_positions_element,
            occupancy,
            true_positions_element,
            loss_config,
        )
        loss_tot = loss_tot + loss
    average_loss = loss_tot / torch.sum(batch_occupancy)
    return average_loss
