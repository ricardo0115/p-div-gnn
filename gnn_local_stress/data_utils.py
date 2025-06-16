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

from typing import Generator

import torch
import torch_geometric as PyG


def slice_batch_gt_and_predictions(
    mesh_graph_batch: PyG.data.Batch,
    prediction: torch.Tensor,
) -> Generator[tuple[PyG.data.Data, torch.Tensor], None, None]:
    for i in range(len(mesh_graph_batch)):
        mesh_graph_i = mesh_graph_batch[i]
        mesh_graph_i_mask = mesh_graph_batch.batch == i
        predicted_local_stress_i = prediction[mesh_graph_i_mask]
        yield mesh_graph_i, predicted_local_stress_i


def slice_batch_predictions(
    batch_graph_prediction: torch.Tensor, batch_indices: torch.Tensor
) -> Generator[torch.Tensor, None, None]:
    n_samples_in_batch = len(torch.unique(batch_indices))
    for i in range(n_samples_in_batch):
        mesh_graph_i_mask = batch_indices == i
        predicted_graph_local_stress = batch_graph_prediction[mesh_graph_i_mask]
        yield predicted_graph_local_stress


def standardize(
    data: torch.Tensor,
    mean: torch.Tensor | tuple[float, float, float],
    std: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    return (data - mean) / std


def unstandardize(
    data: torch.Tensor,
    mean: torch.Tensor | tuple[float, float, float],
    std: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    return (data * std) + mean
