"""Auxiliary functions
"""

import torch
from pytorch_lightning import Callback


def filter_predictions(predicted_points, certainties, certainty_cutoff):
    certainty_mask = filter_certainty(certainties, certainty_cutoff)
    certainties_filtered = certainties[certainty_mask.squeeze()]
    predicted_points_filtered = predicted_points[certainty_mask.squeeze(), :]
    return certainties_filtered, predicted_points_filtered


def filter_certainty(certainty, threshold):
    certainty_mask = certainty > threshold
    return certainty_mask


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA


def radius_graph_custom(pos, r_max, r_min, batch) -> torch.Tensor:
    """Creates edges based on distances between points belonging to the same graph in the batch

    Args:
        pos: tensor of coordinates
        r_max: put no edge if distance is larger than r_max
        r_min: put no edge if distance is smaller than r_min
        batch : info to which graph a node belongs

    Returns:
        index: edges consisting of pairs of node indices
    """
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > r_min)).nonzero().T
    index_mask = index[0] != index[1]
    index = index[:, index_mask]
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index


class InitialCheckpoint(Callback):
    def __init__(self, initial_ckp_path):
        super().__init__()
        self.initial_ckp_path = initial_ckp_path

    def on_train_start(self, trainer, pl_module):
        # Save the initial model checkpoint
        trainer.save_checkpoint(self.initial_ckp_path)
