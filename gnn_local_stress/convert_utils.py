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

import fedoo as fd
import numpy as np
import pyvista as pv
import torch
import torch_geometric as PyG

# Below functions works only with single element-type meshes


def _format_faces_from_pyvista(faces: np.ndarray) -> torch.Tensor:
    n_nodes_per_face = faces[0]
    # Perform copy because original array is not writeable
    # and could produce unexpected tensor behaviors
    mesh_faces = np.copy(faces.reshape(-1, n_nodes_per_face + 1)[:, 1:])
    faces = torch.from_numpy(mesh_faces).t().contiguous()
    return faces


def _format_faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    n_nodes_per_face = faces.shape[1]
    formatted_faces = np.zeros(
        (faces.shape[0], n_nodes_per_face + 1), dtype=np.uint64
    )
    formatted_faces[:, 0] = n_nodes_per_face
    formatted_faces[:, 1:] = faces
    return formatted_faces


def mesh_to_graph(
    mesh: pv.UnstructuredGrid,
    remove_faces: bool = False,
) -> PyG.data.Data:
    # Maybe there is a better way to perform this check is_quad_mesh
    is_quad_mesh = mesh.get_cell(0).type == pv.CellType.QUAD
    faces: torch.Tensor = _format_faces_from_pyvista(mesh.faces)
    graph = PyG.data.Data(pos=torch.from_numpy(mesh.points), face=faces)
    if is_quad_mesh:
        _quad_face_to_edge(graph, remove_faces)
    else:
        face_to_edge = PyG.transforms.FaceToEdge(remove_faces=remove_faces)
        graph = face_to_edge(graph)
    return graph


def _quad_face_to_edge(
    mesh_graph: PyG.data.Data, set_faces_to_none: bool = True
) -> None:
    face_indices = mesh_graph.face
    edge_index = torch.cat(
        [
            face_indices[:2],
            face_indices[1:3],
            face_indices[2:],
            face_indices[::3],
        ],
        dim=1,
    )
    edge_index = PyG.utils.to_undirected(
        edge_index, num_nodes=mesh_graph.num_nodes
    )
    mesh_graph.edge_index = edge_index
    if set_faces_to_none:
        mesh_graph.face = None


def graph_to_mesh(graph: PyG.data.Data) -> pv.PolyData:
    if graph.pos.shape[1] == 2:
        # A 0 column Z array must be added in order to make work format_faces method
        pos = torch.zeros(size=(graph.pos.shape[0], 3))
        pos[:, :2] = graph.pos
        graph.pos = pos
    vertices = graph.pos.detach().cpu().numpy()
    faces = graph.face.detach().t().cpu().numpy()
    faces = _format_faces_to_pyvista(faces)
    return pv.PolyData(vertices, faces)


def format_stress_field_to_fedoo(stress_field: torch.Tensor) -> np.ndarray:
    n_nodes = stress_field.shape[0]
    formatted_stress_field = np.zeros((6, n_nodes))
    formatted_stress_field[[0, 1, 3], :] = stress_field.T
    return formatted_stress_field


def convert_torch_graph_to_fedoo_dataset(graph_mesh: PyG.data.Data) -> fd.Mesh:
    if graph_mesh.is_periodic:
        graph_mesh = PyG.data.Data(
            pos=graph_mesh.pos,
            edge_index=graph_mesh.org_edge_index,
            face=graph_mesh.face,
        )
    pv_mesh = graph_to_mesh(graph_mesh)
    fd_mesh = fd.Mesh.from_pyvista(pv_mesh)
    dataset = fd.DataSet(fd_mesh)
    return dataset
