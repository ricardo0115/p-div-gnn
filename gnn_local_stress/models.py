"""
p-div-gnn
Copyright (C) 2025 Manuel Ricardo GUEVARA GARBAN

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from abc import ABC
from typing import Optional

import torch
import torch_geometric as PyG
from torch.nn import Linear, Sequential
from torch_geometric.nn import (
    LayerNorm,
    MessagePassing,
)


def print_model(
    model: torch.nn.Module,
    data_loader: PyG.loader.DataLoader,
    device: str,
) -> str:
    sample = next(iter(data_loader)).to(device)
    model = model.to(device)
    summary_model = PyG.nn.summary(model, sample)
    return summary_model


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filename: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "mean_pos": model.mean_pos,
        "mean_mean_stress": model.mean_mean_stress,
        "std_mean_stress": model.std_mean_stress,
        "mean_local_stress": model.mean_local_stress,
        "std_pos": model.std_pos,
        "std_local_stress": model.std_local_stress,
        "mean_edge_weight": model.mean_edge_weight,
        "std_edge_weight": model.std_edge_weight,
    }
    torch.save(checkpoint, filename)


def load_model_checkpoint(
    model: torch.nn.Module,
    filename: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    if not torch.cuda.is_available():
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.mean_mean_stress = checkpoint["mean_mean_stress"]
    model.mean_local_stress = checkpoint["mean_local_stress"]
    model.std_mean_stress = checkpoint["std_mean_stress"]
    model.std_local_stress = checkpoint["std_local_stress"]
    model.mean_pos = checkpoint["mean_pos"]
    model.std_pos = checkpoint["std_pos"]
    model.mean_edge_weight = checkpoint["mean_edge_weight"]
    model.std_edge_weight = checkpoint["std_edge_weight"]
    epoch = checkpoint["epoch"]
    return epoch


def load_optimizer_checkpoint(
    optimizer: torch.optim.Optimizer, filename: str
) -> torch.optim.Optimizer:
    checkpoint = torch.load(filename)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return optimizer


class StressFieldBaseModel(torch.nn.Module, ABC):
    def __init__(
        self,
        latent_size: int,
        input_nodes_features_size: int,
        output_nodes_features_size: int,
        mean_pos: Optional[torch.Tensor | tuple[float, float]] = torch.Tensor(
            1
        ),
        std_pos: Optional[torch.Tensor | tuple[float, float]] = torch.Tensor(1),
        mean_mean_stress: Optional[
            torch.Tensor | tuple[float, float]
        ] = torch.Tensor(1),
        std_mean_stress: Optional[
            torch.Tensor | tuple[float, float]
        ] = torch.Tensor(1),
        mean_local_stress: Optional[
            torch.Tensor | tuple[float, float]
        ] = torch.Tensor(1),
        std_local_stress: Optional[
            torch.Tensor | tuple[float, float]
        ] = torch.Tensor(1),
        mean_edge_weight: Optional[
            torch.Tensor | tuple[float, float]
        ] = torch.Tensor(1),
        std_edge_weight: Optional[
            torch.Tensor | tuple[float, float]
        ] = torch.Tensor(1),
    ):
        super().__init__()
        self.latent_size = latent_size
        self.input_nodes_features_size = input_nodes_features_size
        self.output_nodes_features_size = output_nodes_features_size
        self.mean_pos = mean_pos
        self.std_pos = std_pos
        self.mean_mean_stress = mean_mean_stress
        self.std_mean_stress = std_mean_stress
        self.mean_local_stress = mean_local_stress
        self.std_local_stress = std_local_stress
        self.mean_edge_weight = mean_edge_weight
        self.std_edge_weight = std_edge_weight

    def format_node_features(
        self, mesh_graph: PyG.data.Data, scale_data: bool
    ) -> torch.Tensor:
        pos = mesh_graph.pos
        mean_stress = mesh_graph.mean_stress
        nodes_types = mesh_graph.nodes_types
        if scale_data:
            mean_stress = (
                mean_stress - self.mean_mean_stress
            ) / self.std_mean_stress
            pos = (pos - self.mean_pos) / self.std_pos
        x = torch.hstack([mean_stress, pos, nodes_types])
        return x

    def format_edge_features(
        self, mesh_graph: PyG.data.Data, scale_data: bool
    ) -> torch.Tensor:
        edge_attr = mesh_graph.edge_attr
        if scale_data:
            edge_attr = (
                edge_attr - self.mean_edge_weight
            ) / self.std_edge_weight
        return edge_attr

    def to(self, device: str) -> torch.nn.Module:
        _list_attr = [
            "mean_local_stress",
            "std_local_stress",
            "mean_mean_stress",
            "std_mean_stress",
            "mean_pos",
            "std_pos",
            "mean_edge_weight",
            "std_edge_weight",
        ]
        for attr in _list_attr:
            attribute = getattr(self, attr)
            if attribute is not None:
                setattr(self, attr, attribute.to(device))
        return super().to(device)


class Processor(MessagePassing):
    """Message passing."""

    def __init__(
        self,
        latent_size: int,
        input_nodes_features_size: int,
        input_edges_features_size: int,
    ):
        super().__init__(aggr="add")
        self.latent_size = latent_size

        self.edge_net = Sequential(
            Linear(input_edges_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

        self.node_net = Sequential(
            Linear(input_nodes_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

    def forward(self, graph: PyG.data.Data) -> PyG.data.Data:
        edge_index = graph.edge_index
        x = graph.x
        edge_features = graph.edge_attr

        new_node_features = self.propagate(
            edge_index, x=x, edge_attr=edge_features
        )

        row, col = edge_index
        new_edge_features = self.edge_net(
            torch.cat([x[row], x[col], edge_features], dim=-1)
        )

        new_node_features = new_node_features + graph.x
        new_edge_features = new_edge_features + graph.edge_attr

        return PyG.data.Data(
            edge_index=edge_index,
            x=new_node_features,
            edge_attr=new_edge_features,
        )

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)

        return self.edge_net(features)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tmp = torch.cat([aggr_out, x], dim=-1)

        return self.node_net(tmp)


class EncodeProcessDecode(StressFieldBaseModel):
    def __init__(
        self,
        input_edges_features_size: int,
        message_passing_steps: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.message_passing_steps = message_passing_steps
        self.input_edges_features_size = input_edges_features_size
        self.node_encoder = Sequential(
            Linear(self.input_nodes_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

        self.edge_encoder = Sequential(
            Linear(self.input_edges_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

        self.processor = Processor(
            self.latent_size,
            input_nodes_features_size=self.latent_size * 2,
            input_edges_features_size=self.latent_size * 3,
        )

        self.node_decoder = Sequential(
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.output_nodes_features_size),
        )

    def forward(
        self,
        mesh_graph: PyG.data.Data,
        scale_output: bool = True,
        scale_input: bool = True,
    ) -> PyG.data.Data:
        if not torch.any(mesh_graph.mean_stress):
            return PyG.data.Data(
                local_stress=torch.zeros_like(mesh_graph.mean_stress),
                edge_index=mesh_graph.edge_index,
                pos=mesh_graph.pos,
            )
        # Normalize data
        edge_index = mesh_graph.edge_index
        x = self.format_node_features(mesh_graph, scale_input)
        edge_weight = self.format_edge_features(
            mesh_graph, scale_input
        ).unsqueeze(
            1
        )  # Add matrix 1 dimension to edges weights
        node_embedding = self.node_encoder(x)
        edge_embedding = self.edge_encoder(edge_weight)
        latent_graph = PyG.data.Data(
            edge_index=edge_index, x=node_embedding, edge_attr=edge_embedding
        )
        for _ in range(self.message_passing_steps):
            latent_graph = self.processor(latent_graph)

        decoded_nodes = self.node_decoder(latent_graph.x)

        if scale_output:
            decoded_nodes = (
                decoded_nodes * self.std_local_stress
            ) + self.mean_local_stress
        return PyG.data.Data(
            local_stress=decoded_nodes,
            edge_index=mesh_graph.edge_index,
            pos=mesh_graph.pos,
        )
