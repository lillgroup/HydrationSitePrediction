"""Create a html visualization of prediction after clustering
"""

import os
import resource

import hydra
import torch

from hs_prediction.data.dataset import create_dataloaders, create_inference_dataloader
from hs_prediction.models.location.model import create_model
from hs_prediction.utils.auxiliary import set_seed
from hs_prediction.utils.clustering import predict_locations
from inference.visualization.visualize import plot_3d_points_interactive


@hydra.main(
    config_path="../../config/", config_name="location_model", version_base="1.1"
)
def main(config):
    device = config.inference.cuda_ids[0]
    valid_dataloader = create_inference_dataloader(config)
    model = create_model(config)
    # Load checkpoint
    model.load_checkpoint(config.training.resume_path)
    data_loader = valid_dataloader
    model = model.to(device)
    for batch_nr, batch in enumerate(data_loader):
        batch = batch.to(device)
        print(batch_nr)
        certainty_prediction, location_prediction = predict_locations(
            model,
            batch,
            config.evaluation.certainty_cutoff,
            config.evaluation.dst_threshold,
            config.evaluation.cluster_certainty_cutoff,
        )
        base_dir = os.path.join(
            config.general.repo_dir,
            f"images/trajectory_after_training/{batch['key'][0]}",
        )
        os.makedirs(base_dir, exist_ok=True)
        output_file = os.path.join(base_dir, "point_prediction.html")
        water_positions = batch["wat"].pos
        occupancy = batch["wat"].occupancy
        plot_3d_points_interactive(
            location_prediction,
            certainty_prediction,
            water_positions.detach().cpu().numpy(),
            occupancy.detach().cpu().numpy(),
            output_file,
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
