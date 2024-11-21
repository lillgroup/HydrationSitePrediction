import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from hs_prediction.models.utils import get_optimizer, get_scheduler, rbf, unload
from hs_prediction.utils.auxiliary import radius_graph_custom


def filter_edge_indices(edge_indices, max_node):
    """Erase all edges where both source and destination nodes are smaller than max_node.

    Args:
        edge_indices (torch.Tensor): A tensor of shape (2, num_edges) containing the edge indices.
        max_node (int): The threshold node index. Edges where both nodes are less than this index will be removed.

    Returns:
        torch.Tensor: Filtered edge indices tensor.
    """
    # Separate source and destination nodes
    src, dst = edge_indices

    # Create a mask for edges to keep
    mask = (src >= max_node) | (dst >= max_node)

    # Filter edge indices based on the mask
    filtered_edge_indices = edge_indices[:, mask]

    return filtered_edge_indices


def unidirect_edge_indices(edge_indices):
    """Only allow edges where source node is smaller than target node

    Args:
        edge_indices (torch.Tensor): A tensor of shape (2, num_edges) containing the edge indices.
    """
    src, dst = edge_indices
    # Create a mask for edges to keep
    mask = src < dst
    # Filter edge indices based on the mask
    filtered_edge_indices = edge_indices[:, mask]
    return filtered_edge_indices


class GATWithEdgeFeatures(torch.nn.Module):
    def __init__(
        self,
        config,
        input_dim,
        hidden_dim,
        edge_input_dim,
        edge_embedding_dim,
        heads,
        radius_max,
        num_layers=2,
        dropout=0.0,
        output_dim=2,
        ff_num_layers=2,
        hidden_dim_ff=128,
        with_electrostatic_potential=False,
        **kwargs,
    ):
        super(GATWithEdgeFeatures, self).__init__()
        self.config = config
        self.with_electrostatic_potential = with_electrostatic_potential
        if self.with_electrostatic_potential:
            input_dim = input_dim + 22
        self.feature_emb = nn.Linear(input_dim, hidden_dim)
        self.edge_input_dim = edge_input_dim
        self.radius_max = radius_max
        self.gamma = nn.Parameter(torch.Tensor([10.0]), requires_grad=True)

        # First GATConv layer
        self.conv1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_embedding_dim,
        )

        # Additional GATConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_embedding_dim,
                )
            )
        # Define the MLP based on ff_num_layers and hidden_dim_ff
        layers = []
        input_size = hidden_dim * heads
        for _ in range(ff_num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_dim_ff))
            layers.append(nn.ReLU())
            input_size = hidden_dim_ff
        layers.append(nn.Linear(input_size, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.edge_emb = torch.nn.Linear(edge_input_dim, edge_embedding_dim)

    def forward(self, data):
        # Process edge attributes
        protein_pos = data["pro"]["pos"]
        water_pos = data["wat"]["pos"]
        pos_nodes = torch.cat([protein_pos, water_pos], dim=0)
        batch = torch.cat(
            [data["pro"]["batch"], data["wat"]["batch"]],
            dim=0,
        )
        edge_index = to_undirected(
            radius_graph_custom(pos_nodes, self.radius_max, 0.0, batch)
        )  # note that both edge directions exist
        # filter_edges_by_indices(edge_index, filter_out_indices)
        if not self.config.model.inter_atom_edges:
            edge_indices_filtered = filter_edge_indices(edge_index, len(protein_pos))
            edge_index = edge_indices_filtered
        if self.config.model.unidirect_edges:
            edge_indices_filtered = unidirect_edge_indices(edge_index)
            edge_index = edge_indices_filtered
        src, dst = edge_index
        edge_vec = pos_nodes[dst] - pos_nodes[src]
        edge_dis = torch.linalg.norm(edge_vec, dim=-1)
        edge_h = rbf(
            edge_dis,
            min_val=0.0,
            max_val=self.radius_max,
            n_kernels=self.edge_input_dim,
            gamma=self.gamma,
        )
        edge_attr = self.edge_emb(edge_h)
        device = data["pro"]["features"].device
        num_features = data["pro"]["features"].shape[1]
        water_one_hot = torch.zeros(water_pos.shape[0], num_features).to(device)
        water_one_hot[:, -1] = 1
        if self.with_electrostatic_potential:
            electrostatic_potential_rbf = data["wat"]["electrostatic_potential_rbf"]
            water_features = torch.cat(
                [
                    water_one_hot,
                    electrostatic_potential_rbf,
                ],
                dim=1,
            )
            num_protein_atoms = data["pro"]["features"].shape[0]
            protein_features = torch.cat(
                [
                    data["pro"]["features"],
                    torch.zeros(
                        num_protein_atoms, electrostatic_potential_rbf.shape[1]
                    ).to(device),
                ],
                dim=1,
            )
        else:
            water_features = water_one_hot
            protein_features = data["pro"]["features"]
        protein_node_h = self.feature_emb(protein_features)
        water_node_h = self.feature_emb(water_features)
        # Create graph with both protein and water nodes
        h_node = torch.cat([protein_node_h, water_node_h], dim=0)
        h_node = F.elu(self.conv1(h_node, edge_index, edge_attr=edge_attr))
        for conv in self.convs:
            h_node = F.elu(conv(h_node, edge_index, edge_attr=edge_attr))
        h_nodes = self.mlp(h_node)
        h_water = h_nodes[protein_node_h.shape[0] :]
        return h_water


class GATLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
        model,
        optimizer: str = "adam",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler: Optional[str] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super(GATLightningModule, self).__init__()
        self.config = config
        self.water_perturbation_std = config.data.water_perturbation_std
        self.model = model
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_scheduler = optimizer_scheduler
        self.optimizer_scheduler_kwargs = optimizer_scheduler_kwargs
        self.train_outputs = []
        self.valid_outputs = []

    def forward(self, batch, outputs=None):
        thermodynamics = self.model(batch)
        predict_mask = batch["wat"].occupancy >= 0.5
        loss_enthalpy = (
            (thermodynamics[predict_mask, 0] - batch["wat"].enthalpy[predict_mask]) ** 2
        ).mean()
        loss_entropy = (
            (thermodynamics[predict_mask, 1] - batch["wat"].entropy[predict_mask]) ** 2
        ).mean()
        loss_total = torch.tensor(0.0, device=thermodynamics.device)
        if self.config.model.train_enthalpy:
            loss_total = loss_total + loss_enthalpy
        if self.config.model.train_entropy:
            loss_total = loss_total + loss_entropy
        if loss_total is not None and outputs is not None:
            outputs.append(
                {
                    "loss": loss_total,
                    "loss_enthalpy": unload(loss_enthalpy),
                    "loss_entropy": unload(loss_entropy),
                }
            )

        print("loss_total: ", loss_total)
        return loss_total

    @staticmethod
    def epoch_end_metrics(outputs, label: str, stride: int = 1):
        """Compute all metrics at the end of an epoch"""
        losses = [output["loss"] for output in outputs[::stride]]
        enthalpy_losses = [output["loss_enthalpy"] for output in outputs[::stride]]
        entropy_losses = [output["loss_entropy"] for output in outputs[::stride]]
        metrics = {
            f"{label}_loss": torch.tensor(losses).mean().item(),
            f"{label}_loss_enthalpy": torch.tensor(enthalpy_losses).mean().item(),
            f"{label}_loss_entropy": torch.tensor(entropy_losses).mean().item(),
        }
        return metrics

    def training_step(self, batch, batch_idx):
        """Training step (forward pass & loss)"""
        device = batch["wat"].pos.device
        perturbation = 0
        if self.water_perturbation_std > 0:
            perturbation = (
                torch.rand(batch["wat"].pos.shape, device=device)
                * self.water_perturbation_std
            )
        batch["wat"].pos = batch["wat"].pos + perturbation
        loss = self.forward(batch, self.train_outputs)
        return loss

    def on_train_epoch_end(self):
        """Training epoch end (logging)"""
        metrics = self.epoch_end_metrics(self.train_outputs, "train", stride=1)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_outputs = []

    def validation_step(self, batch, batch_idx):
        """Validation step (forward pass & loss)"""
        loss = self.forward(batch, self.valid_outputs)
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end (logging)"""
        metrics = self.epoch_end_metrics(self.valid_outputs, "valid", stride=1)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.valid_outputs = []

    def configure_optimizers(self):
        """Configure optimizer and scheduler for training"""
        optimizer = get_optimizer(self, self.optimizer, **self.optimizer_kwargs)
        if self.optimizer_scheduler is None:
            return optimizer
        else:
            scheduler = get_scheduler(
                optimizer, self.optimizer_scheduler, **self.optimizer_scheduler_kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid_loss",
            }

    def load_checkpoint(self, checkpoint_path):
        assert os.path.exists(
            checkpoint_path
        ), f"resume_path ({checkpoint_path}) does not exist"
        self.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)[
                "state_dict"
            ]
        )
        print(f"Loaded checkpoint from {checkpoint_path}")

    def predict(self, batch):
        """Prediction function to evaluate the model on a new batch"""
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            thermodynamics = self.model(batch)
        return thermodynamics

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        if self.config.training.noise_gradient:
            # Inject gradient noise
            for p in self.parameters():
                if p.grad is not None:
                    noise = (
                        torch.randn_like(p.grad)
                        * self.config.training.noise_gradient_amount
                    )
                    p.grad += noise
        # Perform the optimizer step
        optimizer.step(closure=optimizer_closure)


def create_thermo_model(config: DictConfig) -> GATLightningModule:
    """Create model from configuration"""
    model = GATWithEdgeFeatures(config, **config.model)
    return GATLightningModule(config=config, model=model, **config.training)
