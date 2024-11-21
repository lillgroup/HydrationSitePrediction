from typing import Optional
import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import radius_graph
from torch_geometric.utils import (
    softmax,
    to_undirected,
)
from torch_scatter import scatter_sum
from hs_prediction.models.utils import rbf
from hs_prediction.utils.auxiliary import radius_graph_custom

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class Block(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if activation is not None:
            layers.append(ACTIVATIONS[activation]())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: Optional[str] = "relu",
        final_activation: Optional[str] = None,
    ):
        super().__init__()
        assert num_layers > 1, "MLP must have at least 2 layers"
        self.layers = nn.Sequential(
            Block(input_dim, hidden_dim, dropout, batch_norm, activation),
            *[
                Block(hidden_dim, hidden_dim, dropout, batch_norm, activation)
                for _ in range(num_layers - 2)
            ],
            Block(hidden_dim, output_dim, dropout, batch_norm, final_activation),
        )

    def forward(self, x):
        return self.layers(x)


class EquivariantAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_dim: int,
        num_output_vectors: int = 2,
        radius_max: float = 5.0,
        radius_min: float = 0.2,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: Optional[str] = "relu",
        num_mlp_layers: int = 2,
        **_kwargs
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.radius_max = radius_max
        self.radius_min = radius_min
        self.gamma = nn.Parameter(torch.Tensor([50.0]), requires_grad=True)
        self.gat_conv = GATConv(
            input_dim,
            hidden_dim,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )
        self.number_of_output_vectors = num_output_vectors
        mlp_kwargs = dict(
            dropout=dropout,
            batch_norm=batch_norm,
            activation="relu",
        )
        self.edge_nn = MLP(
            4 * hidden_dim + edge_dim,
            hidden_dim,
            hidden_dim,
            num_layers=3,
            **mlp_kwargs,
            final_activation="relu",
        )
        self.gate_nn = MLP(
            hidden_dim,
            hidden_dim,
            output_dim=self.number_of_output_vectors,
            num_layers=5,
            **mlp_kwargs,
            final_activation="sigmoid",
        )
        self.out_nn = MLP(
            hidden_dim,
            hidden_dim,
            num_layers=5,
            output_dim=self.number_of_output_vectors,
            **mlp_kwargs,
            final_activation=None,
        )

    def forward(self, h, pos, batch, mask, connectivity_mask=False):
        naive_radius_graph_calculation = True
        if naive_radius_graph_calculation:
            edge_index = to_undirected(
                radius_graph_custom(pos, self.radius_max, self.radius_min, batch)
            )  # note that both edge directions exist
        else:
            edge_index_broken = to_undirected(
                radius_graph(pos, self.radius_max, batch, max_num_neighbors=10000)
            )  # note that both edge directions exist
            mask_edges = batch[edge_index_broken[0]] == batch[edge_index_broken[1]]
            edge_index = edge_index_broken[:, mask_edges]
        if connectivity_mask:
            # only connectivity between atoms
            mask_inverse = torch.logical_not(mask)
            edge_choice = torch.all(mask_inverse[edge_index], dim=0)
            edge_index = edge_index[:, edge_choice]
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_dis = torch.linalg.norm(edge_vec, dim=-1)
        # epsilon = 1e-12
        epsilon = 0
        edge_vec_norm = edge_vec / (edge_dis.unsqueeze(-1) + epsilon)
        edge_h = rbf(
            edge_dis,
            min_val=0.0,
            max_val=self.radius_max,
            n_kernels=self.edge_dim,
            gamma=self.gamma,
        )

        # Run GNN
        h_update = self.gat_conv(h, edge_index=edge_index, edge_attr=edge_h)
        # Compute update using attention
        edge_mask = mask[src]
        edge_src, edge_dst = src[edge_mask], dst[edge_mask]
        out_h = self.edge_nn(
            torch.concat(
                [
                    h[edge_src],
                    h_update[edge_src],
                    h[edge_dst],
                    h_update[edge_dst],
                    edge_h[edge_mask],
                ],
                dim=1,
            )
        )  # those are now edge features
        scales = softmax(self.gate_nn(out_h), edge_src) * self.out_nn(out_h)
        update = scatter_sum(
            scales[:, :, None] * edge_vec_norm[edge_mask][:, None, :],
            edge_src,
            dim=0,
            dim_size=h.shape[0],
        )
        new_nodes = pos[mask, None, :] + update[mask]
        new_nodes_reshaped = new_nodes.reshape(-1, 3)
        return new_nodes_reshaped, h_update
