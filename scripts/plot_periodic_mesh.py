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

from tempfile import NamedTemporaryFile

import gmsh
import numpy as np
import pyvista as pv
from microgen.mesh import is_periodic

from gnn_local_stress import datasets

WINDOW_SIZE = (2000, 2000)
OFF_SCREEN = True
MESH_LINE_WIDTH = 2
DASHED_MESH_LINE_WIDTH = 4
POINT_SIZE = 20


def _create_dashed_line(
    start: np.ndarray,
    end: np.ndarray,
    dash_length: float,
    gap_length: float,
) -> list[pv.Line]:
    """Creates a dashed line between two points with specified dash and gap lengths."""
    direction = end - start
    distance = np.linalg.norm(direction)
    direction = direction / distance  # Normalize direction

    segments = []
    current_position = start
    while np.linalg.norm(current_position - start) < distance:
        next_position = current_position + direction * dash_length
        if np.linalg.norm(next_position - start) > distance:
            next_position = end  # Ensure we don’t exceed the final point
        segments.append(pv.Line(current_position, next_position))
        current_position = next_position + direction * gap_length
        if np.linalg.norm(current_position - start) >= distance:
            break
    return segments


def _plot_periodic_edges(
    mesh: pv.UnstructuredGrid, plotter: pv.Plotter, offset_value: float = 0.1
) -> pv.Plotter:
    points_2d = mesh.points[:, :-1]
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
        point_mask[np.lexsort((points_2d[point_mask].T))]
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
    corner_points = np.array(
        [
            left_lower_corner,
            left_upper_corner,
            right_lower_corner,
            right_upper_corner,
        ]
    ).squeeze()

    lines = []

    offset = offset_value * max(
        mesh.points[:, 0].max() - mesh.points[:, 0].min(),
        mesh.points[:, 1].max() - mesh.points[:, 1].min(),
    )
    for (
        left_idx,
        right_idx,
    ) in zip(
        left_side_points_mask,
        right_side_points_mask,
    ):
        left_point = mesh.points[left_idx]
        right_point = mesh.points[right_idx]
        # Define virtual points just outside the mesh bounds
        left_virtual_point = left_point - np.array([offset, 0, 0])
        right_virtual_point = right_point + np.array([offset, 0, 0])

        # Create lines from actual boundary points to their virtual counterparts
        lines.append(pv.Line(left_point, left_virtual_point))
        lines.append(pv.Line(right_point, right_virtual_point))

    for lower_idx, upper_idx in zip(
        lower_side_points_mask,
        upper_side_points_mask,
    ):
        lower_point = mesh.points[lower_idx]
        upper_point = mesh.points[upper_idx]
        # Define virtual points just outside the mesh bounds
        lower_virtual_point = lower_point - np.array([0, offset, 0])
        upper_virtual_point = upper_point + np.array([0, offset, 0])
        # Create lines from actual boundary points to their virtual counterparts
        lines.append(pv.Line(lower_point, lower_virtual_point))
        lines.append(pv.Line(upper_point, upper_virtual_point))

    # Compute periodic edges of corners
    left_lower_corner_point = mesh.points[left_lower_corner]

    left_lower_corner_virtual_point = left_lower_corner_point - np.array(
        [offset, offset, 0]
    )
    lines.append(
        pv.Line(left_lower_corner_point, left_lower_corner_virtual_point)
    )

    left_upper_corner_point = mesh.points[left_upper_corner]

    left_upper_corner_virtual_point = left_upper_corner_point - np.array(
        [offset, -offset, 0]
    )
    lines.append(
        pv.Line(left_upper_corner_point, left_upper_corner_virtual_point)
    )
    right_lower_corner_point = mesh.points[right_lower_corner]

    right_lower_corner_virtual_point = right_lower_corner_point - np.array(
        [-offset, offset, 0]
    )
    lines.append(
        pv.Line(right_lower_corner_point, right_lower_corner_virtual_point)
    )

    right_upper_corner_point = mesh.points[right_upper_corner]

    right_upper_corner_virtual_point = right_upper_corner_point - np.array(
        [-offset, -offset, 0]
    )
    lines.append(
        pv.Line(right_upper_corner_point, right_upper_corner_virtual_point)
    )
    dash_length = 0.5
    gap_length = 0.3
    segmented_lines = []
    for line in lines:
        segmented_lines.extend(
            _create_dashed_line(
                line.points[0],
                line.points[1],
                dash_length,
                gap_length,
            )
        )
    # Combine all lines into a single mesh
    edges_mesh = pv.MultiBlock(segmented_lines).combine()

    # Plot the mesh and periodic edges
    plotter.add_mesh(
        edges_mesh,
        color="purple",
        line_width=DASHED_MESH_LINE_WIDTH,
        label="Periodic Edges",
    )
    return plotter


def _center_plotter_camera(
    plotter: pv.Plotter, mesh: pv.UnstructuredGrid
) -> None:
    # Adjust the camera to fit the mesh exactly
    plotter.camera_position = "xy"
    plotter.camera.SetFocalPoint(mesh.center)
    plotter.camera.SetPosition(mesh.center[0], mesh.center[1], 1)
    plotter.camera.SetViewUp([0, 1, 0])
    # Reset the camera to fit the mesh
    plotter.reset_camera()

    # Set parallel projection to avoid perspective distortion
    plotter.camera.ParallelProjectionOn()


def hole_plate_mesh(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
) -> pv.UnstructuredGrid:
    hole_mesh_refinement_size = (
        global_mesh_refinement_size / hole_refinement_factor
    )
    gmsh.initialize()

    # Define 4 points of the plate
    square_points = [
        gmsh.model.geo.add_point(0, 0, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(width, 0, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(width, height, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(0, height, 0, global_mesh_refinement_size),
    ]

    square_lines = [
        # Trace 4 lines corresponding to each tag
        gmsh.model.geo.add_line(square_points[0], square_points[1]),
        gmsh.model.geo.add_line(square_points[1], square_points[2]),
        gmsh.model.geo.add_line(square_points[2], square_points[3]),
        gmsh.model.geo.add_line(square_points[3], square_points[0]),
    ]

    # Define center of the circle

    center_x, center_y = hole_center
    center_point = gmsh.model.geo.add_point(
        center_x, center_y, 0, global_mesh_refinement_size
    )

    # Define Two points at the beginning and at the end of the circle
    cp1 = gmsh.model.geo.add_point(
        center_x - radius, center_y, 0, hole_mesh_refinement_size
    )
    cp2 = gmsh.model.geo.add_point(
        center_x + radius, center_y, 0, hole_mesh_refinement_size
    )
    # Add circle arcs between points
    # init_point_tag, center_point_tag, end_point_tag, arc_tag
    circle_arc0 = gmsh.model.geo.add_circle_arc(cp1, center_point, cp2)
    circle_arc1 = gmsh.model.geo.add_circle_arc(cp2, center_point, cp1)

    surface_plate = gmsh.model.geo.add_curve_loop(square_lines)
    surface_hole = gmsh.model.geo.add_curve_loop([circle_arc0, circle_arc1])

    mesh_surface = gmsh.model.geo.add_plane_surface(
        [surface_plate, surface_hole]
    )

    gmsh.model.geo.synchronize()

    # Meshing algoritm can be changed
    gmsh.model.mesh.set_algorithm(
        dim=2, tag=mesh_surface, val=5
    )  # 5 best quality 8 for quadrangles
    # gmsh.model.mesh.setRecombine(2, mesh_surface)  # To have quadrangles
    mesh_dim = 2
    gmsh.model.mesh.generate(mesh_dim)

    with NamedTemporaryFile(suffix=".msh", delete=True) as file:
        filename = file.name
        gmsh.write(filename)
        gmsh.finalize()
        shape = pv.read_meshio(filename)
        shape = shape.extract_cells_by_type(
            pv.CellType.TRIANGLE
        )  # Remove line elements
    assert is_periodic(shape.points[:, :-1]), "Mesh is not periodic"
    return shape


def plot_mesh_with_periodic_edges(
    mesh: pv.UnstructuredGrid, offset_value: float
) -> None:
    pl = pv.Plotter(window_size=WINDOW_SIZE, off_screen=OFF_SCREEN)
    node_labels = datasets.compute_node_labels(mesh)
    # Add External boundary points
    pl.add_mesh(
        mesh,
        color="w",
        show_edges=True,
        line_width=MESH_LINE_WIDTH,
        opacity=0.5,
    )
    pl.add_points(
        mesh.points[node_labels == datasets.NodeType.EXTERNAL_BOUNDARY],
        color="r",
        point_size=POINT_SIZE,
        label="External Surface",
    )
    pl = _plot_periodic_edges(plotter=pl, mesh=mesh, offset_value=offset_value)
    _center_plotter_camera(pl, mesh)
    pl.camera.zoom("tight")

    # pl.add_legend()
    if OFF_SCREEN:
        pl.save_graphic("periodic.pdf", raster=False)
    # pl.show()


def plot_mesh_with_node_labels(mesh: pv.UnstructuredGrid) -> None:
    pl = pv.Plotter(window_size=WINDOW_SIZE, off_screen=OFF_SCREEN)
    pl.add_mesh(mesh, color="w", show_edges=True, line_width=2, opacity=0.5)
    node_labels = datasets.compute_node_labels(mesh)
    # Add External boundary points
    pl.add_points(
        mesh.points[node_labels == datasets.NodeType.INTERNAL],
        color="blue",
        point_size=POINT_SIZE,
        label="Internal",
    )
    pl.add_points(
        mesh.points[node_labels == datasets.NodeType.EXTERNAL_BOUNDARY],
        color="r",
        point_size=POINT_SIZE,
        label="External surface",
    )
    pl.add_points(
        mesh.points[node_labels == datasets.NodeType.INTERNAL_BOUNDARY],
        color="orange",
        point_size=POINT_SIZE,
        label="Internal surface",
    )
    _center_plotter_camera(pl, mesh)
    pl.add_legend(border=True, bcolor="w")
    # pl.camera.zoom("tight")
    pl.show()
    if OFF_SCREEN:
        pl.save_graphic("labels.pdf", raster=False)


def _plot_node_labels(
    plotter: pv.Plotter, mesh: pv.UnstructuredGrid
) -> pv.Plotter:
    node_labels = datasets.compute_node_labels(mesh)
    # Add External boundary points
    plotter.add_points(
        mesh.points[node_labels == datasets.NodeType.INTERNAL],
        color="blue",
        point_size=POINT_SIZE,
        label="Internal",
    )
    plotter.add_points(
        mesh.points[node_labels == datasets.NodeType.EXTERNAL_BOUNDARY],
        color="r",
        point_size=POINT_SIZE,
        label="External surface",
    )
    plotter.add_points(
        mesh.points[node_labels == datasets.NodeType.INTERNAL_BOUNDARY],
        color="orange",
        point_size=POINT_SIZE,
        label="Internal surface",
    )
    return plotter


def plot_node_labels_and_periodic_edges(mesh: pv.UnstructuredGrid) -> None:
    pl = pv.Plotter(window_size=WINDOW_SIZE, off_screen=OFF_SCREEN)
    pl.add_mesh(mesh, color="w", show_edges=True, line_width=2, opacity=0.5)
    pl = _plot_node_labels(pl, mesh)
    pl = _plot_periodic_edges(plotter=pl, mesh=mesh)
    _center_plotter_camera(pl, mesh)
    pl.add_legend(border=True, bcolor="w", background_opacity=0.9)
    pl.camera.zoom("tight")
    pl.camera.zoom(0.98)
    if OFF_SCREEN:
        pl.save_graphic("labels_and_periodic.pdf", raster=False)
    else:
        pl.show()


def main() -> None:
    width = 100
    height = width
    hole_center_radius = 20
    hole_center_pos = (width // 2, height // 2)
    global_mesh_refinement_size = 10
    hole_refinement_factor = 2
    mesh = hole_plate_mesh(
        width,
        height,
        hole_center_radius,
        hole_center_pos,
        hole_refinement_factor,
        global_mesh_refinement_size,
    )
    plot_node_labels_and_periodic_edges(mesh)


if __name__ == "__main__":
    main()
