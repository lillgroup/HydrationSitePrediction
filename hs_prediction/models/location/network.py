import os

import torch
from torch import nn

from hs_prediction.models.location.layers import MLP, EquivariantAttention


class EGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        radius_max: float = 8.0,
        radius_min: float = 0.2,
        num_mlp_layers: int = 2,
        edge_embedding_dim: int = 32,
        num_output_vectors: int = 2,
        num_weight_layers: int = 3,
        **_kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_embedding_dim = edge_embedding_dim
        self.num_output_vectors = num_output_vectors
        self.feature_emb = nn.Linear(input_dim, hidden_dim)
        self.initial_placement_nn = EquivariantAttention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_embedding_dim,
            num_output_vectors=self.num_output_vectors,
            radius_max=radius_max,
            radius_min=radius_min,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            num_mlp_layers=num_mlp_layers,
        )

        self.layers = nn.ModuleList(
            [
                EquivariantAttention(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    edge_dim=edge_embedding_dim,
                    num_output_vectors=1,
                    radius_max=radius_max,
                    radius_min=radius_min,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    activation=activation,
                    num_mlp_layers=num_mlp_layers,
                )
                for _ in range(num_layers)
            ]
        )
        self.h_to_weight_nn = MLP(
            hidden_dim, hidden_dim, 1, num_layers=num_weight_layers
        )

    def forward(self, data, save_dir=None):
        mask_put_wat = data["pro"]["put_wat"]
        pro_node_h = self.feature_emb(data["pro"].features)
        device = pro_node_h.device
        water_initial, _ = self.initial_placement_nn(
            pro_node_h, data["pro"].pos, data["pro"].batch, mask_put_wat
        )
        if save_dir is not None:
            torch.save(
                data["pro"].pos[mask_put_wat].detach().cpu(),
                os.path.join(save_dir, "placement.pt"),
            )
            torch.save(
                water_initial.detach().cpu(),
                os.path.join(save_dir, "water_pred_layer_0.pt"),
            )
            torch.save(
                data["wat"].pos.detach().cpu(), os.path.join(save_dir, "true_water.pt")
            )

        num_features = data["pro"].features.shape[1]
        num_wat = water_initial.shape[0]
        water_one_hot = torch.zeros(num_wat, num_features).to(device)
        water_one_hot[:, -1] = 1
        water_node_h = self.feature_emb(water_one_hot)
        # Create graph with both protein and water nodes
        h = torch.cat([pro_node_h, water_node_h], dim=0)
        pos = torch.cat(
            [data["pro"].pos, water_initial], dim=0
        )  # the position of water is unknown. we could place them at atom positions initially

        batch = torch.cat(
            [
                data["pro"].batch,
                torch.repeat_interleave(
                    data["pro"].batch[mask_put_wat], self.num_output_vectors
                ),
                # water_batch,
            ],
            dim=0,
        )
        mask = torch.zeros_like(batch, dtype=torch.bool)
        mask[data["pro"].num_nodes :] = True

        # Propagate through layers
        for i in range(self.num_layers):
            updated_pos, h_update = self.layers[i](h, pos, batch, mask)
            pos[mask] = updated_pos
            if save_dir is not None:
                torch.save(
                    updated_pos.detach().cpu(),
                    os.path.join(save_dir, f"water_pred_layer_{i+1}.pt"),
                )
                # if i == self.num_layers - 1:
                #    logits = self.h_to_weight_nn(h_update[data["pro"].num_nodes :])
                #    certainty = torch.sigmoid(logits)
                #    torch.save(
                #        certainty.detach().cpu(),
                #        os.path.join(save_dir, f"water_pred_layer_certainty.pt"),
                #    )
        # Calculate weights
        logits = self.h_to_weight_nn(h_update[data["pro"].num_nodes :])
        certainty = torch.sigmoid(logits)
        water_prediction_pos = pos[data["pro"].num_nodes :]
        batch_water = batch[data["pro"].num_nodes :]
        # calculate weights from h_update
        return (certainty, water_prediction_pos, batch_water)
