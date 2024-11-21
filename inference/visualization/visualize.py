""" Visualize prediction results
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pymol
import torch
from pymol import cmd


def load_sessions_as_frames(session_files):
    """
    Load multiple PSE files into a single object as different states.

    Args:
    session_files (list): List of paths to .pse session files.
    """
    # Initialize target object
    target_object = "combined_session"
    cmd.create(target_object, "none")

    for index, session_file in enumerate(session_files):
        # Load session file
        cmd.load(session_file)
        objects = cmd.get_names("objects")
        for obj in objects:
            cmd.create(target_object, obj, 1, index + 1)  # Append as new state
        # Clean up the session before loading the next one
        cmd.delete("all")
    cmd.save("combined_output.pse", target_object)


def create_pymol_object(protein_pdb_path, predicted_coords, true_coords, output_path):
    # Initialize PyMOL in headless mode
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    # Load the protein PDB file
    cmd.load(protein_pdb_path, "protein")
    # Use suspend updates to optimize performance
    cmd.set("suspend_updates", "on")
    # Handle predicted coordinates more efficiently
    for i, coord in enumerate(predicted_coords):
        cmd.pseudoatom(
            object=f"predicted_point_{i}",
            pos=tuple(coord),
            color="red",
            vdw=0.1,
        )
    # Handle true coordinates
    for i, coord in enumerate(true_coords):
        cmd.pseudoatom(
            object=f"true_point_{i}", pos=tuple(coord), color="green", vdw=0.1
        )
    # Group pseudoatoms for better management
    cmd.group("predicted_points", "predicted_point_*")
    cmd.group("true_points", "true_point_*")
    # Adjust settings for better visualization
    cmd.show("spheres", "predicted_points")
    cmd.show("spheres", "true_points")
    cmd.set("sphere_scale", 0.1, "predicted_points")
    cmd.set("sphere_scale", 0.1, "true_points")
    # Resume updates
    cmd.set("suspend_updates", "off")
    # Save the session to the specified output path
    cmd.save(output_path)
    cmd.delete("all")


def pymol_creation(
    data_set_path, protein_key, predicted_coords, true_coords, output_path
):
    path = os.path.join(data_set_path, protein_key)
    protein_pdb_path = os.path.join(path, "protein.pdb")
    create_pymol_object(protein_pdb_path, predicted_coords, true_coords, output_path)


def execute_specific_pymol_commands(cmd, sample_dir):
    cmd.set_view(
        (
            -0.875431955,
            -0.189145356,
            0.444797158,
            -0.307837784,
            0.927657783,
            -0.211396068,
            -0.372634530,
            -0.321987659,
            -0.870326698,
            -0.000000000,
            0.000000000,
            -71.351150513,
            18.583997726,
            16.758867264,
            17.179933548,
            56.253791809,
            86.448509216,
            -20.000000000,
        )
    )
    cmd.set("ray_opaque_background", "off")
    cmd.set("ray_trace_frames", 1)
    for i in range(1, 9):  # Adjust the range if needed
        cmd.frame(i)
        cmd.png(os.path.join(sample_dir, f"frame_{i}.png"), dpi=300)


def create_pymol_objects_in_states(
    protein_pdb_path, predictions, true_coords_list, sample_dir
):

    output_path = os.path.join(sample_dir, f"subsequent_predictions.pse")
    # Initialize PyMOL in headless mode
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()

    # Load the protein PDB file once, it will be common across all states
    cmd.load(protein_pdb_path, "protein")

    # Use suspend updates to optimize performance
    cmd.set("suspend_updates", "on")

    # Assume predictions and true_coords_list are lists of lists, each sublist a different state
    for state_index, (predicted_coords, true_coords) in enumerate(
        zip(predictions, true_coords_list), start=1
    ):
        # Create a single object for predicted points for this state
        for i, coord in enumerate(predicted_coords):
            cmd.pseudoatom(
                object=f"predicted_points",  # A single object for all predicted points
                pos=tuple(coord),
                color="red",
                vdw=0.1,
                state=state_index,  # Specify the state
                resi=i + 1,  # Optional: Use resi to identify individual points
            )

        # Create a single object for true points for this state
        for i, coord in enumerate(true_coords):
            cmd.pseudoatom(
                object=f"true_points",  # A single object for all true points
                pos=tuple(coord),
                color="green",
                vdw=0.1,
                state=state_index,  # Specify the state
                resi=i + 1,  # Optional: Use resi to identify individual points
            )

    # Group pseudoatoms for better management
    cmd.group("predicted_points_group", "predicted_points")
    cmd.group("true_points_group", "true_points")

    # Adjust settings for better visualization across all states
    cmd.show("spheres", "true_points")
    cmd.hide("nonbonded", "true_points")
    cmd.set("sphere_scale", 2.0, "true_points")
    cmd.color("forest", "true_points")
    cmd.color("blue", "protein")

    # Resume updates
    cmd.set("suspend_updates", "off")

    execute_specific_pymol_commands(cmd, sample_dir)

    # Save the session to the specified output path
    cmd.save(output_path)
    cmd.delete("all")


def plot_3d_points(predicted_coords, true_coords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Convert predicted and true coordinates to numpy arrays for easier handling
    predicted_coords = np.array(predicted_coords)
    true_coords = np.array(true_coords)
    # Plot predicted points in red
    ax.scatter(
        predicted_coords[:, 0],
        predicted_coords[:, 1],
        predicted_coords[:, 2],
        c="red",
        label="Predicted Points",
    )
    # Plot true points in green
    ax.scatter(
        true_coords[:, 0],
        true_coords[:, 1],
        true_coords[:, 2],
        c="green",
        label="True Points",
    )
    # Setting labels and title
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")
    ax.set_zlabel("Z Coordinates")
    ax.set_title("3D Plot of Predicted and True Points")
    ax.legend()
    # Save the plot to a file
    plt.show()
    plt.close()


def plot_3d_points_interactive(
    predicted_coords, certainty, true_coords, occupancy, output_file
):
    # Convert list to numpy array for easier indexing
    predicted_coords = np.array(predicted_coords)
    true_coords = np.array(true_coords)
    certainty = np.array(
        certainty
    )  # Assuming this is an array of the same length as the coords
    certainty_scaled = torch.sigmoid(torch.tensor(certainty)).numpy()

    # Create a Plotly figure
    fig = go.Figure()
    scaling_factor = 10

    hovertext = [f"Certainty: {c}" for c in certainty]
    # Add predicted points to the plot
    fig.add_trace(
        go.Scatter3d(
            x=predicted_coords[:, 0],
            y=predicted_coords[:, 1],
            z=predicted_coords[:, 2],
            mode="markers",
            marker=dict(size=certainty_scaled * 5, color="red", opacity=0.5),
            name="Predicted Points",
            hovertext=hovertext,
            hoverinfo="text",
        )
    )

    hovertext = [f"Occupancy: {c}" for c in occupancy]
    # Add true points to the plot
    fig.add_trace(
        go.Scatter3d(
            x=true_coords[:, 0],
            y=true_coords[:, 1],
            z=true_coords[:, 2],
            mode="markers",
            marker=dict(size=occupancy * scaling_factor, color="green", opacity=0.5),
            name="True Points",
            hovertext=hovertext,
            hoverinfo="text",
        )
    )

    # Set plot layout
    fig.update_layout(
        title="3D Plot of Predicted and True Points with Certainty",
        scene=dict(
            xaxis_title="X Coordinates",
            yaxis_title="Y Coordinates",
            zaxis_title="Z Coordinates",
        ),
        legend_title="Point Type",
    )
    # Save the plot to an HTML file
    fig.write_html(output_file)


def create_pymol_protein_with_predictions(
    protein_pdb_path, predicted_coords, enthalpies, entropies, true_waters, output_path
):
    """Create a PyMOL object from the protein and predicted points with associated enthalpy and entropy values.

    Args:
        protein_pdb_path (str): Path to the protein PDB file.
        predicted_coords (torch.Tensor): Predicted coordinates of waters, shape (num_points, 3).
        enthalpies (torch.Tensor): Associated enthalpy values, shape (num_points,).
        entropies (torch.Tensor): Associated entropy values, shape (num_points,).
        true_waters: true coordinates of the water molecules
        output_path (str): Path to save the results.
    """
    # Initialize PyMOL in headless mode
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()

    # Load the protein PDB file once, it will be common across all states
    cmd.load(protein_pdb_path, "protein")

    # Use suspend updates to optimize performance
    cmd.set("suspend_updates", "on")

    # Handle predicted coordinates for this state
    for ind, coord in enumerate(predicted_coords):
        enthalpy = enthalpies[ind].item()
        entropy = entropies[ind].item()

        cmd.pseudoatom(
            object=f"predicted_point_{ind}",
            pos=tuple(coord),
            color="red",
            vdw=0.1,
        )
        # Label the pseudoatom with the enthalpy and entropy values
        cmd.label(
            f"predicted_point_{ind}",
            f'"Enthalpy: {enthalpy:.2f}, Entropy: {entropy:.2f}"',
        )
    if true_waters is not None:
        for ind, coord in enumerate(true_waters):
            cmd.pseudoatom(
                object=f"true_waters_{ind}",
                pos=tuple(coord),
                color="green",
                vdw=0.1,
            )
        cmd.group("true_waters", "true_waters_*")
        cmd.show("spheres", "true_waters_*")
        cmd.set("sphere_scale", 0.1, "true_waters_*")

    # Group pseudoatoms for better management
    cmd.group("predicted_points", "predicted_point_*")

    # Adjust settings for better visualization across all states

    cmd.show("spheres", "predicted_point_*")
    cmd.set("sphere_scale", 0.1, "predicted_point_*")
    # Hide labels initially
    cmd.hide("labels", "predicted_point_*")
    # Show labels only on hover
    cmd.set("label_color", "white", "predicted_point_*")
    cmd.set("label_size", -0.5, "predicted_point_*")
    # Resume updates
    cmd.set("suspend_updates", "off")
    # Save the session to the specified output path
    cmd.save(output_path)
    cmd.delete("all")
