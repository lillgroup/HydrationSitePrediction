"""Create predictions for watsite data and experimental data (xray), create joint pymol visualization"""

import os
from os.path import join

import hydra
import pymol
import torch
from pymol import cmd

from hs_prediction.data.dataset import create_dataloaders
from hs_prediction.models.location.model import create_model
from hs_prediction.utils.clustering import predict_locations


def save_results(path_to_pdb, predictions, waters, save_dir):
    torch.save(path_to_pdb, join(save_dir, "pdb_path.pt"))
    torch.save(
        torch.tensor(predictions),
        os.path.join(save_dir, f"predictions.pt"),
    )
    torch.save(
        torch.tensor(waters),
        os.path.join(save_dir, f"waters.pt"),
    )


def create_pymol_points(predictions, waters, postfix):
    for predicted_coords, true_coords in zip(predictions, waters):
        # Handle predicted coordinates for this state
        for i, coord in enumerate(predicted_coords):
            cmd.pseudoatom(
                object=f"pred_{postfix}_{i}",
                pos=tuple(coord),
                color="red",
                vdw=0.1,
            )
        # Handle true coordinates for this state
        for i, coord in enumerate(true_coords):
            cmd.pseudoatom(
                object=f"water_{postfix}_{i}",
                pos=tuple(coord),
                color="green",
                vdw=0.1,
            )
        # Group pseudoatoms for better management
        cmd.group(f"pred_{postfix}", f"pred_{postfix}_*")
        cmd.group(f"water_{postfix}", f"water_{postfix}_*")

    # Adjust settings for better visualization across all states
    cmd.show("spheres", f"pred_{postfix}_*")
    cmd.show("spheres", f"water_{postfix}_*")
    cmd.set("sphere_scale", 0.1, f"pred_{postfix}_*")
    cmd.set("sphere_scale", 0.1, f"water_{postfix}_*")


def create_joint_pymol(dir_watsite, dir_experiments, output_path):
    # Initialize PyMOL in headless mode
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    # Use suspend updates to optimize performance
    cmd.set("suspend_updates", "on")
    for dir, postfix in [(dir_watsite, "watsite"), (dir_experiments, "experiment")]:
        pdb_path = join(dir, "pdb_path.pt")
        predictions_path = join(dir, "predictions.pt")
        waters_path = join(dir, "waters.pt")
        path_to_pdb = torch.load(pdb_path, weights_only=True)
        predictions = torch.load(predictions_path, weights_only=True)
        waters = torch.load(waters_path,weights_only=True)
        # Load the protein PDB file once, it will be common across all states
        cmd.load(path_to_pdb, f"protein_{postfix}")
        create_pymol_points([predictions], [waters], postfix)

    cmd.create(
        f"water_pred_protein_watsite",
        f"protein_watsite or water_watsite or pred_watsite",
    )
    cmd.align("water_pred_protein_watsite", "protein_experiment")
    # Resume updates
    cmd.set("suspend_updates", "off")
    # Save the session to the specified output path
    cmd.save(output_path)
    cmd.delete("all")


def save_predictions(config, element_names, save_dir):
    device = config.inference.cuda_ids[0]
    train_dataloader, valid_dataloader, _, _ = create_dataloaders(config)
    model = create_model(config)
    model = model.to(device)
    model.load_checkpoint(config.training.resume_path)
    for data_loader in [train_dataloader, valid_dataloader]:
        for ind, batch in enumerate(data_loader):
            if batch.key[0] in element_names:
                print("Processing sample ", batch.key[0])
                batch = batch.to(device)
                key = batch.key[0]
                save_path = join(save_dir, f"{batch.key[0]}")
                os.makedirs(save_path, exist_ok=True)
                certainty_prediction, location_prediction = predict_locations(
                    model,
                    batch,
                    config.evaluation.certainty_cutoff,
                    config.evaluation.dst_threshold,
                    config.evaluation.cluster_certainty_cutoff,
                )
                water_positions = batch["wat"].pos
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                path_to_pdb = join(config.data.dataset_path, key, "protein.pdb")
                save_results(
                    path_to_pdb, location_prediction, water_positions, save_path
                )


@hydra.main(
    config_path="../../config/", config_name="location_model", version_base="1.1"
)
def main(config):
    if config.data.name == "watsite":
        element_list = ["6ek3", "4u58", "1uyg", "4zsm", "6qtq"]
    elif config.data.name == "xray_water":
        element_list = ["Q460N3", "P52293", "P07900", "P56817", "P43254"]
    else:
        raise ValueError(
            "Please change the data set used to 'watsite' or 'xray_water' in the config file"
        )
    save_dir = os.path.join(
        config.general.repo_dir,
        "images/trajectory_after_training/watsite_vs_experiment/",
    )
    save_predictions(config, element_list, save_dir)
    ###############################################
    # merge the two predictions into one pymol file
    dir_watsite = os.path.join(save_dir, "6ek3")
    dir_experiments = os.path.join(save_dir, "Q460N3")
    output_path = os.path.join(save_dir, f"joint_6ek3_Q460N3.pse")
    create_joint_pymol(dir_watsite, dir_experiments, output_path)


if __name__ == "__main__":
    main()
