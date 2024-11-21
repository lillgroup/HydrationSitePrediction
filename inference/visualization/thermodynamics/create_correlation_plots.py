"""Evaluate results for entropy and enthalpy
"""

import os
import resource

import hydra
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from scipy.stats import linregress, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from hs_prediction.data.dataset import create_dataloaders
from hs_prediction.models.thermodynamics.model_entropy import create_thermo_model
from hs_prediction.utils.auxiliary import set_seed


def fit_linear_regression(true_values: torch.Tensor, predicted_values: torch.Tensor):
    """
    Fits a linear regression model to the true and predicted values.

    Args:
        true_values (torch.Tensor): Tensor containing true values.
        predicted_values (torch.Tensor): Tensor containing predicted values.

    Returns:
        LinearRegression: Fitted linear regression model.
    """
    # Convert tensors to numpy arrays
    true_values_np = true_values.cpu().numpy().reshape(-1, 1)
    predicted_values_np = predicted_values.cpu().numpy().reshape(-1, 1)
    # Fit linear regression model
    model = LinearRegression()
    model.fit(predicted_values_np, true_values_np)
    return model


def refine_predictions(model, predicted_values: torch.Tensor):
    """
    Refines the predicted values using the fitted linear regression model.

    Args:
        model (LinearRegression): Fitted linear regression model.
        predicted_values (torch.Tensor): Tensor containing predicted values.

    Returns:
        torch.Tensor: Refined predicted values.
    """
    # Convert tensor to numpy array
    predicted_values_np = predicted_values.cpu().numpy().reshape(-1, 1)

    # Refine predictions
    refined_predictions_np = model.predict(predicted_values_np)

    # Convert back to tensor
    refined_predictions = torch.tensor(
        refined_predictions_np, device=predicted_values.device
    )

    return refined_predictions


def evaluate_mse(true_values: torch.Tensor, refined_predictions: torch.Tensor):
    """
    Evaluates the Mean Squared Error (MSE) between true values and refined predictions.

    Args:
        true_values (torch.Tensor): Tensor containing true values.
        refined_predictions (torch.Tensor): Tensor containing refined predicted values.

    Returns:
        float: Mean Squared Error (MSE).
    """
    # Convert tensors to numpy arrays
    true_values_np = true_values.cpu().numpy()
    refined_predictions_np = refined_predictions.cpu().numpy()

    # Calculate MSE
    mse = mean_squared_error(true_values_np, refined_predictions_np)

    return mse


def compute_and_print_slope(
    true_values: torch.Tensor, predicted_values: torch.Tensor, label: str
):
    """
    Computes and prints the slope of the linear regression between true and predicted values.

    Args:
        true_values (torch.Tensor): Tensor containing true values.
        predicted_values (torch.Tensor): Tensor containing predicted values.
        label (str): Label for the type of data (e.g., 'enthalpy' or 'entropy').
    """
    slope, intercept, r_value, p_value, std_err = linregress(
        predicted_values.cpu().numpy(), true_values.cpu().numpy()
    )
    print(f"Slope of the linear regression for {label}: {slope}")
    print(f"Intercept of the linear regression for {label}: {intercept}")
    print(f"R-squared of the linear regression for {label}: {r_value**2}")


def create_scatter_plot(
    true_values: torch.Tensor,
    predicted_values: torch.Tensor,
    title: str,
    output_file: str,
):
    """
    Creates a scatter plot for true vs predicted values with a dotted line through the origin.

    Args:
        true_values (torch.Tensor): Tensor containing true values.
        predicted_values (torch.Tensor): Tensor containing predicted values.
        title (str): Title of the plot.
        output_file (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_values.cpu().numpy(), true_values.cpu().numpy(), alpha=0.5)
    # Add a dotted line through the origin
    min_val = min(true_values.min().item(), predicted_values.min().item())
    max_val = max(true_values.max().item(), predicted_values.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def create_hexbin_plot(
    true_values: torch.Tensor,
    predicted_values: torch.Tensor,
    title: str,
    output_file: str,
    line_through_origin=False,
    xlabel="Predicted Values",
    ylabel="True Values",
):
    """
    Creates a hexbin plot for true vs predicted values.

    Args:
        true_values (torch.Tensor): Tensor containing true values.
        predicted_values (torch.Tensor): Tensor containing predicted values.
        title (str): Title of the plot.
        output_file (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hexbin(
        predicted_values.cpu().numpy(),
        true_values.cpu().numpy(),
        gridsize=100,
        cmap="Blues",
        mincnt=1,
        norm=LogNorm(),
    )
    if line_through_origin:
        # Add a line with a slope of 1 (y = x line)
        min_val = min(true_values.min().item(), predicted_values.min().item())
        max_val = max(true_values.max().item(), predicted_values.max().item())
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label="Counts")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


@hydra.main(
    config_path="../../../config/", config_name="thermo_model", version_base="1.1"
)
def main(config):
    device = config.inference.cuda_ids[0]
    train_dataloader, valid_dataloader, means, stds = create_dataloaders(config)
    is_train = True
    if is_train:
        data_loader = train_dataloader
        post_fix = "train"
    else:
        data_loader = valid_dataloader
        post_fix = "valid"
    model = create_thermo_model(config)
    model = model.to(device)
    if config.training.resume_path is not None:
        model.load_checkpoint(config.training.resume_path)
    enthalpy_predicted = []
    entropy_predicted = []
    enthalpy_truth = []
    entropy_truth = []
    occupancies = []
    for batch_nr, batch in enumerate(data_loader):
        batch = batch.to(device)
        print(batch_nr)
        with torch.no_grad():
            thermodynamics_predicted = model.model(batch)
        predict_mask = batch["wat"].occupancy >= 0.5
        occupancies.append(batch["wat"].occupancy[predict_mask])
        predicted_enthalpy = thermodynamics_predicted[predict_mask, 0]
        predicted_entropy = thermodynamics_predicted[predict_mask, 1]
        true_enthalpy = batch["wat"].enthalpy[predict_mask]
        true_entropy = batch["wat"].entropy[predict_mask]
        enthalpy_predicted.append(predicted_enthalpy)
        entropy_predicted.append(predicted_entropy)
        enthalpy_truth.append(true_enthalpy)
        entropy_truth.append(true_entropy)
        print(((predicted_enthalpy - true_enthalpy) ** 2).mean())

    occupancies_tensor = torch.cat(occupancies)
    enthalpy_predicted_tensor = torch.cat(enthalpy_predicted)
    enthalpy_truth_tensor = torch.cat(enthalpy_truth)
    base_dir = os.path.join(config.general.repo_dir, "images/thermodynamics/")
    os.makedirs(base_dir, exist_ok=True)
    title = "Enthalpy Correlation Plot"
    file_format = "eps"
    output_file = os.path.join(base_dir, f"enthalpy_{post_fix}.{file_format}")
    create_hexbin_plot(
        enthalpy_truth_tensor.cpu(), enthalpy_predicted_tensor.cpu(), title, output_file
    )
    pearson_corr_enthalpy, _ = pearsonr(
        enthalpy_truth_tensor.cpu().numpy(), enthalpy_predicted_tensor.cpu().numpy()
    )
    print("pearson r correlation for enthalpy: ", pearson_corr_enthalpy)
    entropy_predicted_tensor = torch.cat(entropy_predicted)
    entropy_truth_tensor = torch.cat(entropy_truth)
    pearson_corr_entropy, _ = pearsonr(
        entropy_truth_tensor.cpu().numpy(), entropy_predicted_tensor.cpu().numpy()
    )
    print("pearson r correlation for entropy: ", pearson_corr_entropy)
    title = "Entropy Correlation Plot"
    output_file = os.path.join(base_dir, f"entropy_{post_fix}.{file_format}")
    create_hexbin_plot(
        entropy_truth_tensor.cpu(), entropy_predicted_tensor.cpu(), title, output_file
    )
    ################################
    # plot squared entropy difference vs occupancy
    ################################
    entropy_sqr_diff = (entropy_predicted_tensor - entropy_truth_tensor) ** 2
    pearson_corr_entropy, _ = pearsonr(
        entropy_sqr_diff.cpu().numpy(), occupancies_tensor.cpu().numpy()
    )
    print("pearson r correlation for entropy diff vs occupancy: ", pearson_corr_entropy)
    title = "Correlation Plot: Entropy Squared Difference vs Occupancy "
    output_file = os.path.join(
        base_dir, f"entropydiff_vs_occupancy_{post_fix}.{file_format}"
    )
    create_hexbin_plot(
        entropy_sqr_diff.cpu(),
        occupancies_tensor.cpu(),
        title,
        output_file,
        xlabel="Occupancy",
        ylabel="Squared Entropy Difference",
    )
    ################################
    # plot true entropy vs occupancy
    ################################
    pearson_corr_entropy, _ = pearsonr(
        entropy_truth_tensor.cpu().numpy(), occupancies_tensor.cpu().numpy()
    )
    print(
        "pearson r correlation for true entropy diff vs occupancy: ",
        pearson_corr_entropy,
    )
    title = "Correlation Plot: Entropy vs Occupancy "
    output_file = os.path.join(
        base_dir, f"entropy_vs_occupancy_{post_fix}.{file_format}"
    )
    create_hexbin_plot(
        entropy_truth_tensor.cpu(),
        occupancies_tensor.cpu(),
        title,
        output_file,
        xlabel="Occupancy",
        ylabel="Entropy",
    )

    ####################################
    # try linear regression and find slope:
    # Compute and print slope for enthalpy
    compute_and_print_slope(
        enthalpy_truth_tensor, enthalpy_predicted_tensor, "enthalpy"
    )
    # Compute and print slope for entropy
    compute_and_print_slope(entropy_truth_tensor, entropy_predicted_tensor, "entropy")
    enthalpy_model = fit_linear_regression(
        enthalpy_truth_tensor, enthalpy_predicted_tensor
    )
    entropy_model = fit_linear_regression(
        entropy_truth_tensor, entropy_predicted_tensor
    )
    # Refine predictions
    refined_enthalpy_predictions_regression = refine_predictions(
        enthalpy_model, enthalpy_predicted_tensor
    )
    refined_entropy_predictions_regression = refine_predictions(
        entropy_model, entropy_predicted_tensor
    )
    ##############################################################################
    title = "Enthalpy prediction vs ground truth"
    output_file = os.path.join(base_dir, f"enthalpy_{post_fix}_refined.{file_format}")
    create_hexbin_plot(
        enthalpy_truth_tensor.cpu(),
        refined_enthalpy_predictions_regression.cpu().flatten(),
        title,
        output_file,
    )
    title = "Entropy prediction vs ground truth"
    output_file = os.path.join(base_dir, f"entropy_{post_fix}_refined.{file_format}")
    create_hexbin_plot(
        entropy_truth_tensor.cpu(),
        refined_entropy_predictions_regression.cpu().flatten(),
        title,
        output_file,
    )
    compute_and_print_slope(
        enthalpy_truth_tensor,
        refined_enthalpy_predictions_regression.cpu().flatten(),
        "enthalpy",
    )
    # Compute and print slope for entropy
    compute_and_print_slope(
        entropy_truth_tensor,
        refined_entropy_predictions_regression.cpu().flatten(),
        "entropy",
    )
    ##############################################################################
    # Evaluate MSE
    enthalpy_mse = evaluate_mse(enthalpy_truth_tensor, enthalpy_predicted_tensor)
    print(f"Enthalpy MSE for trained model: {enthalpy_mse}")
    enthalpy_mse = evaluate_mse(
        enthalpy_truth_tensor, refined_enthalpy_predictions_regression
    )
    print(f"Enthalpy MSE for regression refinement: {enthalpy_mse}")
    entropy_mse = evaluate_mse(entropy_truth_tensor, entropy_predicted_tensor)
    print(f"Entropy MSE for trained model: {entropy_mse}")
    entropy_mse = evaluate_mse(
        entropy_truth_tensor, refined_entropy_predictions_regression
    )
    print(f"Entropy MSE for regression refinement: {entropy_mse}")
    ###############################################


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    # torch.use_deterministic_algorithms(True)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
