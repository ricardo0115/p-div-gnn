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

import numpy as np
import pytest
import pyvista as pv
from torch_geometric.data import Data

from gnn_local_stress import convert_utils


def are_mesh_and_graph_equivalent(mesh: pv.PolyData, graph_mesh: Data) -> bool:
    assert mesh
    assert graph_mesh

    expected_n_nodes = mesh.n_points
    expected_n_faces = mesh.n_faces_strict
    expected_nodes_coords = mesh.points
    # Not sure about this cause mesh can have multi element types
    # and the number of nodes per faces could change given the type element
    n_nodes_per_face = mesh.faces[0]
    expected_faces_data = mesh.faces.reshape(-1, n_nodes_per_face + 1)[:, 1:]
    graph_n_nodes = graph_mesh.num_nodes
    graph_n_faces = graph_mesh.face.size(-1)
    graph_nodes_coords = graph_mesh.pos.numpy()
    graph_faces_data = graph_mesh.face.t().numpy()
    # Act
    graph_mesh = convert_utils.mesh_to_graph(
        mesh,
        remove_faces=False,
    )
    # Assert
    assert graph_n_nodes == expected_n_nodes
    assert graph_n_faces == expected_n_faces
    assert np.array_equal(graph_nodes_coords, expected_nodes_coords)
    assert np.array_equal(graph_faces_data, expected_faces_data)
    return True


# Arrange
# Try adding multitype element meshes in this parametrize
@pytest.mark.parametrize(
    "mesh", [pv.Sphere().triangulate(), pv.Box().triangulate()]
)
def test_mesh_to_graph_given_mesh_must_convert_into_corresponding_graph(
    mesh: pv.PolyData,
) -> None:
    # Act
    graph_mesh = convert_utils.mesh_to_graph(
        mesh,
        remove_faces=False,
    )
    # Assert
    assert are_mesh_and_graph_equivalent(mesh=mesh, graph_mesh=graph_mesh)


@pytest.mark.parametrize(
    "mesh", [pv.Sphere().triangulate(), pv.Box().triangulate()]
)
def test_graph_to_mesh_must_return_equivalent_mesh_after_conversion(
    mesh: pv.PolyData,
) -> None:
    # Arrange
    graph_mesh = convert_utils.mesh_to_graph(
        mesh,
        remove_faces=False,
    )
    # Act
    converted_mesh = convert_utils.graph_to_mesh(graph_mesh)
    # Assert
    assert are_mesh_and_graph_equivalent(
        mesh=converted_mesh, graph_mesh=graph_mesh
    )
