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

from enum import IntEnum

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyvista as pv
import torch
import torch_geometric as PyG
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset

from gnn_local_stress.convert_utils import mesh_to_graph


class NodeType(IntEnum):
    INTERNAL_BOUNDARY = -1
    INTERNAL = 0
    EXTERNAL_BOUNDARY = 1


def compute_periodic_graph(graph: PyG.data.Data) -> PyG.data.Data:
    points_2d = graph.pos[:, :-1].numpy()
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)
    mesh_indices = np.arange(len(points_2d))
    left_side_points_mask = np.where(points_2d[:, 0] == min_x)[0]
    right_side_points_mask = np.where(points_2d[:, 0] == max_x)[0]
    upper_side_points_mask = np.where(points_2d[:, 1] == max_y)[0]
    lower_side_points_mask = np.where(points_2d[:, 1] == min_y)[0]

    # Sort mask points
    (
        left_side_points_mask,
        right_side_points_mask,
        upper_side_points_mask,
        lower_side_points_mask,
    ) = [
        torch.from_numpy(point_mask[np.lexsort((points_2d[point_mask].T))])
        for point_mask in [
            left_side_points_mask,
            right_side_points_mask,
            upper_side_points_mask,
            lower_side_points_mask,
        ]
    ]
    left_lower_corner = mesh_indices[
        np.logical_and(*(points_2d == (min_x, min_y)).T)
    ]
    left_upper_corner = mesh_indices[
        np.logical_and(*(points_2d == (min_x, max_y)).T)
    ]
    right_lower_corner = mesh_indices[
        np.logical_and(*(points_2d == (max_x, min_y)).T)
    ]
    right_upper_corner = mesh_indices[
        np.logical_and(*(points_2d == (max_x, max_y)).T)
    ]
    corner_points = torch.from_numpy(
        np.array(
            [
                left_lower_corner,
                left_upper_corner,
                right_lower_corner,
                right_upper_corner,
            ]
        ).squeeze()
    )

    row, col = graph.edge_index
    n_row = torch.cat(
        (
            row,
            left_side_points_mask,
            right_side_points_mask,
            lower_side_points_mask,
            upper_side_points_mask,
            corner_points,
        )
    )
    n_col = torch.cat(
        (
            col,
            right_side_points_mask,
            left_side_points_mask,
            upper_side_points_mask,
            lower_side_points_mask,
            corner_points.flip(dims=[0]),
        )
    )
    n_edge_index = torch.vstack([n_row, n_col]).long()
    # Fill edge_attr with zeros for new connections

    edge_attr = torch.zeros(n_edge_index.shape[1])
    edge_attr[: graph.num_edges] = graph.edge_attr
    return PyG.data.Data(
        edge_index=n_edge_index,
        pos=graph.pos,
        edge_attr=edge_attr,
        face=graph.face,
        org_edge_index=graph.edge_index,
    ).coalesce()


def _regions_must_be_inverted(
    hole_plate_mesh: pv.PolyData | pv.UnstructuredGrid,
    external_boundary_nodes_mask: np.ndarray,
) -> bool:
    bounds_2d = hole_plate_mesh.bounds[:-2]
    compare_point = hole_plate_mesh.points[external_boundary_nodes_mask][
        :, :-1
    ][0]
    return not any([coord in bounds_2d for coord in compare_point])


def compute_node_labels(
    hole_plate_mesh: pv.PolyData | pv.UnstructuredGrid,
) -> npt.NDArray[np.int_]:
    """
    Given a hole plate mesh returns a vector sized hole_plate_mesh.n_points
    corresponding to NodeType enum class type
    """
    boundary_regions = (
        hole_plate_mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            manifold_edges=False,
            feature_edges=False,
        )
        .connectivity()
        .cell_data_to_point_data()
    )
    region_ids: npt.NDArray[np.int_] = boundary_regions.point_data["RegionId"]
    n_regions = len(set(region_ids))

    # Check that only 2 regions has been extracted
    assert (
        n_regions == 2
    ), f"Expected 2 regions, found {n_regions} for the given mesh"

    boundary_regions_mask = boundary_regions.point_data["vtkOriginalPointIds"]
    # RegionId of external boundary nodes is 0
    external_boundary_nodes_mask = boundary_regions_mask[region_ids == 0]

    # RegionId of internal boundary nodes is 1
    internal_boundary_nodes_mask = boundary_regions_mask[region_ids == 1]

    # Sometimes RegionId can be inverted due to the size of regions
    # External boundary nodes MUST intersect with mesh.bounds if not, it means
    # That we should invert them
    if _regions_must_be_inverted(hole_plate_mesh, external_boundary_nodes_mask):
        external_boundary_nodes_mask, internal_boundary_nodes_mask = (
            internal_boundary_nodes_mask,
            external_boundary_nodes_mask,
        )

    node_types = np.full(
        hole_plate_mesh.n_points, NodeType.INTERNAL, dtype=np.int_
    )
    node_types[external_boundary_nodes_mask] = NodeType.EXTERNAL_BOUNDARY
    node_types[internal_boundary_nodes_mask] = NodeType.INTERNAL_BOUNDARY
    return node_types


def _compute_node_distances_as_edge_weights(mesh_graph: Data) -> torch.Tensor:
    distances = (
        mesh_graph.pos[mesh_graph.edge_index[0]]
        - mesh_graph.pos[mesh_graph.edge_index[1]]
    )
    edge_distances = torch.linalg.vector_norm(distances, dim=1)
    return edge_distances


def _init_op_div_matrix(
    mesh_data: dict[str, np.ndarray],
    hyperelastic: bool = False,
) -> torch.sparse.Tensor:
    op_div_matrix_data = torch.FloatTensor(mesh_data["op_div_matrix_data"])
    op_div_matrix_col_indices = torch.LongTensor(
        mesh_data["op_div_matrix_col_indices"]
    )
    op_div_matrix_row_indices = torch.LongTensor(
        mesh_data["op_div_matrix_row_indices"]
    )
    op_div_matrix_shape = torch.Size(mesh_data["op_div_matrix_shape"])
    op_div_matrix_indices = torch.vstack(
        (op_div_matrix_row_indices, op_div_matrix_col_indices)
    )
    # Create a torch sparse tensor
    op_div_matrix = torch.sparse_coo_tensor(
        op_div_matrix_indices,
        op_div_matrix_data,
        op_div_matrix_shape,
        dtype=torch.float32,
    ).coalesce()
    return op_div_matrix


def von_mises_stress(
    mean_stress_x: torch.Tensor,
    mean_stress_y: torch.Tensor,
    mean_stress_xy: torch.Tensor,
) -> np.ndarray | float:
    return np.sqrt(
        0.5
        * (
            (mean_stress_x - mean_stress_y) ** 2
            + mean_stress_x**2
            + mean_stress_y**2
            + 6 * mean_stress_xy**2
        )
    )


class MeshStressFieldDatasetInMemory(InMemoryDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: T.BaseTransform = None,
        periodic_graph: bool = True,
    ) -> None:
        super().__init__(None, transform, None, None)
        mesh_filenames = dataframe["mesh_filename"]
        data_filenames = dataframe["data_filename"]
        self.dataframe = dataframe

        graphs: list[Data] = []
        for (
            mesh_filename,
            data_filename,
        ) in zip(
            mesh_filenames,
            data_filenames,
        ):
            mesh = pv.get_reader(mesh_filename).read()
            graph = mesh_to_graph(mesh)
            graph.edge_attr = _compute_node_distances_as_edge_weights(
                graph
            ).float()
            if periodic_graph:
                graph = compute_periodic_graph(graph)
            graph.is_periodic = periodic_graph
            mesh_data = np.load(data_filename)
            stress_field = torch.from_numpy(mesh_data["stress_field"]).float()
            mean_sigma_x, mean_sigma_y, mean_sigma_xy = mesh_data["mean_stress"]
            mean_stress = torch.ones(stress_field.shape)
            mean_stress = mean_stress * torch.Tensor(
                (mean_sigma_x, mean_sigma_y, mean_sigma_xy)
            )
            graph.pos = graph.pos[
                :, :2
            ].float()  # Remove Z coordinate and cast to float32
            graph.mean_stress = mean_stress
            graph.local_stress = stress_field
            graph.op_div_matrix = _init_op_div_matrix(mesh_data)
            graph.von_mises = von_mises_stress(  # type float
                mean_sigma_x, mean_sigma_y, mean_sigma_xy
            )
            graph.surfaces_nodes_for_div = torch.unsqueeze(
                torch.from_numpy(mesh_data["node_labels"]),
                1,
            )
            graph.nodes_types = graph.surfaces_nodes_for_div
            graphs.append(graph)

        data, slices = self.collate(graphs)
        self.mean_pos = data.pos.mean()
        self.std_pos = data.pos.std()
        self.mean_mean_stress = data.mean_stress.mean()
        self.std_mean_stress = data.mean_stress.std()
        self.mean_local_stress = data.local_stress.mean()
        self.std_local_stress = data.local_stress.std()
        self.mean_edge_weight = data.edge_attr.mean()
        self.std_edge_weight = data.edge_attr.std()
        self.data = data
        self.slices = slices

    @property
    def raw_file_names(self) -> None: ...

    @property
    def processed_file_names(self) -> None: ...

    @property
    def has_download(self) -> bool:
        return False

    @property
    def has_process(self) -> bool:
        return False

    def download(self) -> None: ...

    def process(self) -> None: ...
