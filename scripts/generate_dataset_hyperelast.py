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

import multiprocessing
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, NamedTuple, Sequence

import fedoo as fd
import gmsh
import numpy as np
import pandas as pd
import pyvista as pv
import scipy
import simcoon
from fire import Fire
from microgen.mesh import is_periodic
from tqdm.contrib.concurrent import process_map

from gnn_local_stress import datasets


class MeanStress(NamedTuple):
    xx: float
    yy: float
    xy: float


@dataclass
class DatasetParameters:
    mesh_filename: str
    data_filename: str
    mean_stress_x: float
    mean_stress_y: float
    mean_stress_xy: float
    mean_strain_x: float
    mean_strain_y: float
    mean_strain_xy: float
    hole_plate_center_x: float
    hole_plate_center_y: float
    hole_plate_radius: float
    plate_width: float
    plate_height: float
    global_mesh_refinement_size: float
    hole_mesh_refinement_factor: float
    n_nodes: int
    n_elements: int
    seed: int


def _compute_mean_stress_operator(mesh: fd.Mesh) -> np.ndarray:
    fd.Assembly.delete_memory()
    mesh.reset_interpolation()
    op_to_get_mean = (
        mesh._get_gaussian_quadrature_mat().data
        @ mesh._get_node2gausspoint_mat()
    ) / mesh.bounding_box.volume
    mesh.reset_interpolation()
    fd.Assembly.delete_memory()
    return op_to_get_mean


def _compute_op_div_matrix(mesh: fd.Mesh) -> scipy.sparse.csr_matrix:
    # mesh.nodes = mesh.nodes[:-2, :]
    fd.Assembly.delete_memory()
    mesh.remove_isolated_nodes()
    mesh.reset_interpolation()
    dummy_wf = fd.weakform.StressEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(0, 0)
    )
    assembly = fd.Assembly.create(dummy_wf, mesh)
    op_div = mesh._get_gausspoint2node_mat() @ assembly._get_assembled_operator(
        assembly.space.op_div_u()
    )
    fd.Assembly.delete_memory()
    return op_div.tocoo()


@dataclass
class MechanicalFields:
    stress_field_per_node: np.ndarray
    strain_field_per_node: np.ndarray
    op_div_matrix: scipy.sparse.coo_matrix
    mean_stress: MeanStress


def hole_plate_mesh_quad(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
) -> fd.Mesh:
    hole_mesh_refinement_size = (
        global_mesh_refinement_size / hole_refinement_factor
    )
    gmsh.initialize()
    gmsh.model.add("PlateWithHole")

    # Rectangle
    rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, width, height)

    # Hole
    cx, cy = hole_center
    hole_tag = gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)

    # Synchronize to get boundary entities
    gmsh.model.occ.synchronize()

    # Get boundary edges of the disk BEFORE cutting
    hole_edges = gmsh.model.getBoundary([(2, hole_tag)], recursive=True)

    # Cut disk from rectangle
    gmsh.model.occ.cut(
        [(2, rect_tag)], [(2, hole_tag)], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()

    # Recombine surfaces to create quadrangles
    for surface in gmsh.model.getEntities(2):
        gmsh.model.mesh.setRecombine(2, surface[1])

    # Global mesh size
    gmsh.model.mesh.setSize(
        gmsh.model.getEntities(0), global_mesh_refinement_size
    )

    # Refine near the hole using the pre-captured edges
    gmsh.model.mesh.setSize(hole_edges, hole_mesh_refinement_size)

    gmsh.model.mesh.generate(2)
    with NamedTemporaryFile(suffix=".msh", delete=True) as file:
        filename = file.name
        gmsh.write(filename)
        gmsh.finalize()
        shape = pv.read_meshio(filename)
        shape = shape.extract_cells_by_type(
            pv.CellType.QUAD
        )  # Remove line elements
    assert is_periodic(shape.points[:, :-1])
    return fd.Mesh.from_pyvista(shape).as_2d()


def hole_plate_mesh(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
) -> fd.Mesh:
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
    return fd.Mesh.from_pyvista(shape).as_2d()


def compute_mechanical_fields_dirichlet(
    mesh: fd.Mesh,
    eps_xx: float,
    eps_yy: float,
    gamma_xy: float,
    young_modulus: float = 1e5,
    poisson_ratio: float = 0.3,
) -> MechanicalFields:

    F = simcoon.eR_to_F(
        simcoon.v2t_strain([eps_xx, eps_yy, 0, gamma_xy, 0, 0]), np.eye(3)
    )
    det_F = np.linalg.det(F)
    deformed_volume = mesh.bounding_box.volume * det_F
    fd.Assembly.delete_memory()
    # --------------- Pre-Treatment --------------------------------------------------------
    space = fd.ModelingSpace("2Dplane")
    type_el = mesh.elm_type
    center = mesh.nearest_node(mesh.bounding_box.center)

    strain_nodes = mesh.add_virtual_nodes(2)
    C10 = 1.5  # mu = 2*(C10+C01)
    C01 = 0
    # kappa = 0.5e2  # =2/D1
    kappa = 10
    props = np.array([2 * C10, kappa])
    material = fd.constitutivelaw.Simcoon("NEOHC", props)
    material.use_elastic_lt = False

    # Assembly
    wf = fd.weakform.StressEquilibrium(material)
    # wf.fbar = True
    assemb = fd.Assembly.create(wf, mesh, type_el)

    # Type of problem
    pb = fd.problem.NonLinear(assemb, nlgeom=True)

    grad_U = F - np.eye(3)
    bc_periodic = fd.constraint.PeriodicBC(
        [
            [strain_nodes[0], strain_nodes[0]],
            [strain_nodes[1], strain_nodes[1]],
        ],
        [["DispX", "DispY"], ["DispX", "DispY"]],
        dim=2,
    )

    pb.bc.add(bc_periodic)

    pb.bc.add("Dirichlet", center, "Disp", 0, name="center")

    pb.bc.remove("_Strain")
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[0]],
        "DispX",
        grad_U[0, 0],
        start_value=0,
        name="_Strain",
    )  # dU/dx
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[1]],
        "DispY",
        grad_U[1, 1],
        start_value=0,
        name="_Strain",
    )  # dV/dy
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[0]],
        "DispY",
        grad_U[0, 1],
        start_value=0,
        name="_Strain",
    )  # dU/dy
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[1]],
        "DispX",
        grad_U[1, 0],
        start_value=0,
        name="_Strain",
    )  # dV/dx

    pb.apply_boundary_conditions()

    pb.set_nr_criterion(max_subiter=5, err0=None, tol=1e-3)
    pb.nlsolve(dt=0.02, update_dt=True, interval_output=0.05, print_info=0)
    res = pb.get_results(assemb, ["Disp", "Stress", "Strain"], "Node")
    # mean_stress

    stress_field_per_node = pb.get_results(assemb, "Stress", "Node")["Stress"]
    strain_field_per_node = pb.get_results(assemb, "Strain", "Node")["Strain"]
    # Filter XX, YY, XY
    xx_yy_xy_indices = [0, 1, 3]
    stress_field_per_node = stress_field_per_node[xx_yy_xy_indices]
    strain_field_per_node = strain_field_per_node[xx_yy_xy_indices]

    # Compute mean stress
    mean_stress = [
        mesh.integrate_field(res["Stress", i], type_field="Node")
        / deformed_volume
        for i in xx_yy_xy_indices
    ]
    # Debug
    # for component in ["XX", "YY", "XY", "vm"]:
    #    pb.get_results(assemb, "Stress", "Node").plot(
    #        "Stress", component=component
    #    )
    # Remove virtual nodes added by fedoo
    op_div_matrix = _compute_op_div_matrix(assemb.current.mesh)
    return MechanicalFields(
        stress_field_per_node=stress_field_per_node.T[:-2, :],
        strain_field_per_node=strain_field_per_node.T[:-2, :],
        mean_stress=MeanStress(*mean_stress),
        op_div_matrix=op_div_matrix,
    )


def _compute_random_center_hole_points(
    rng: np.random.Generator,
    plate_height: float,
    plate_width: float,
    padding_factor: float,
    n_samples: int,
    min_radius: float = 5,
) -> np.ndarray:
    padding = plate_width * padding_factor
    padding_spacement = min_radius + (2 * padding)
    hole_center_x_values = rng.uniform(
        low=padding_spacement,
        high=plate_width - padding_spacement,
        size=n_samples,
    )
    hole_center_y_values = rng.uniform(
        low=padding_spacement,
        high=plate_height - padding_spacement,
        size=n_samples,
    )
    hole_center_values = np.stack([hole_center_x_values, hole_center_y_values])
    return hole_center_values


def _compute_random_hole_radius(
    rng: np.random.Generator,
    center_points: np.ndarray,
    plate_height: float,
    plate_width: float,
    padding_factor: float,
    min_radius: float = 5,
) -> np.ndarray:
    padding = plate_height * padding_factor
    center_x, center_y = center_points
    # Find max radius values for generating random hole radius values
    max_random_radius_values = np.min(
        np.stack(
            [
                ((plate_height - padding) - center_y),
                center_y - padding,
                (plate_width - padding) - center_x,
                center_x - padding,
            ]
        ),
        axis=0,
    )
    n_points = center_points.shape[1]
    return rng.uniform(
        low=np.repeat(min_radius + padding, n_points),
        high=max_random_radius_values,
        size=n_points,
    )


def _compute_random_dataset_combinations(
    random_generator: np.random.Generator,
    plate_width: float,
    plate_height: float,
    padding_factor: float,
    strain_range: tuple[float, float],
    global_mesh_refinement_range: tuple[float, float],
    hole_mesh_refinement_factor_range: tuple[float, float],
    n_samples: int,
) -> Iterable[np.ndarray]:
    mean_strain_values = random_generator.uniform(
        low=strain_range[0],
        high=strain_range[1],
        size=(n_samples, 3),
    )
    # mean_strain_values[:, 2] *= 2  # GammaXY = 2EPSXY
    center_points = _compute_random_center_hole_points(
        plate_width=plate_width,
        plate_height=plate_height,
        padding_factor=padding_factor,
        n_samples=n_samples,
        rng=random_generator,
    )
    radius_values = _compute_random_hole_radius(
        center_points=center_points,
        plate_height=plate_height,
        plate_width=plate_width,
        padding_factor=padding_factor,
        rng=random_generator,
    )

    mean_strain_x, mean_strain_y, mean_strain_xy = mean_strain_values.T
    center_points_x, center_points_y = center_points
    global_refinement_values = random_generator.uniform(
        low=global_mesh_refinement_range[0],
        high=global_mesh_refinement_range[1],
        size=n_samples,
    )
    hole_mesh_refinement_factor_values = random_generator.uniform(
        low=hole_mesh_refinement_factor_range[0],
        high=hole_mesh_refinement_factor_range[1],
        size=n_samples,
    )
    return (
        mean_strain_x,
        mean_strain_y,
        mean_strain_xy,
        center_points_x,
        center_points_y,
        radius_values,
        global_refinement_values,
        hole_mesh_refinement_factor_values,
    )


def _generate_and_save_samples_parallel(
    data: np.ndarray,
    plate_width_height: float,
    meshes_folder: Path,
    local_fields_folder: Path,
    seed: int,
) -> pd.DataFrame:
    (
        strain_x,
        strain_y,
        strain_xy,
        center_point_x,
        center_point_y,
        radius,
        global_mesh_refinement,
        hole_mesh_refinement_factor,
        index,
    ) = data
    mesh: fd.Mesh = hole_plate_mesh(
        width=plate_width_height,
        height=plate_width_height,
        hole_center=(center_point_x, center_point_y),
        radius=radius,
        global_mesh_refinement_size=global_mesh_refinement,
        hole_refinement_factor=hole_mesh_refinement_factor,
    )

    # Bug fix workaround, this prevents nodes ids being changed after surface
    # extraction
    mesh = mesh.to_pyvista().extract_surface()
    mesh = fd.Mesh.from_pyvista(mesh)
    try:
        mechanical_fields = compute_mechanical_fields_dirichlet(
            mesh, eps_xx=strain_x, eps_yy=strain_y, gamma_xy=strain_xy
        )
    except Exception as e:
        print(f"An error has occurred {e} of type {type(e)}")
        parameters = f"{strain_x=}, {strain_y=}, {strain_xy=}, {center_point_x=}, {center_point_y=}, {radius=}, {global_mesh_refinement=}, {hole_mesh_refinement_factor=}, {index=},"
        print(f"Example case with parameters {parameters} failed")
        return

    pv_mesh: pv.PolyData = mesh.to_pyvista().extract_surface()
    sample_filename = f"hole_plate_mesh_{str(int(index))}"
    mesh_filename = (meshes_folder / f"{sample_filename}.vtk").as_posix()
    data_filename = (local_fields_folder / f"{sample_filename}.npz").as_posix()
    gen_parameters = DatasetParameters(
        mesh_filename=mesh_filename,
        data_filename=data_filename,
        mean_stress_x=mechanical_fields.mean_stress.xx,
        mean_stress_y=mechanical_fields.mean_stress.yy,
        mean_stress_xy=mechanical_fields.mean_stress.xy,
        mean_strain_x=strain_x,
        mean_strain_y=strain_y,
        mean_strain_xy=strain_xy,
        hole_plate_radius=radius,
        hole_plate_center_x=center_point_x,
        hole_plate_center_y=center_point_y,
        plate_width=plate_width_height,
        plate_height=plate_width_height,
        n_nodes=pv_mesh.n_points,
        n_elements=pv_mesh.n_cells,
        global_mesh_refinement_size=global_mesh_refinement,
        hole_mesh_refinement_factor=hole_mesh_refinement_factor,
        seed=seed,
    )
    # Format mean stress
    pv_mesh.save(mesh_filename)
    node_labels = datasets.compute_node_labels(pv_mesh)
    np.savez(
        data_filename,
        stress_field=mechanical_fields.stress_field_per_node,
        mean_stress=np.array(mechanical_fields.mean_stress),
        mean_strain=np.array((strain_x, strain_y, strain_xy)),
        op_div_matrix_data=mechanical_fields.op_div_matrix.data,
        op_div_matrix_col_indices=mechanical_fields.op_div_matrix.col,
        op_div_matrix_row_indices=mechanical_fields.op_div_matrix.row,
        op_div_matrix_shape=mechanical_fields.op_div_matrix.shape,
        node_labels=node_labels,
    )

    dataframe = pd.json_normalize(asdict(gen_parameters))
    return dataframe


def generate_and_save_samples(
    mean_strain_x_values: Sequence[float],
    mean_strain_y_values: Sequence[float],
    mean_strain_xy_values: Sequence[float],
    center_points_x_values: Sequence[tuple],
    center_points_y_values: Sequence[tuple],
    radius_values: Sequence[float],
    global_mesh_refinement_values: Sequence[float],
    hole_mesh_refinement_values: Sequence[float],
    plate_width_height: float,
    dataset_folder: Path,
    seed: int,
    max_workers: int,
) -> pd.DataFrame:
    meshes_folder = dataset_folder / Path("meshes/")
    local_fields_folder = dataset_folder / Path("fields/")
    meshes_folder.mkdir(parents=True, exist_ok=False)
    local_fields_folder.mkdir(parents=True, exist_ok=False)

    function = partial(
        _generate_and_save_samples_parallel,
        plate_width_height=plate_width_height,
        meshes_folder=meshes_folder,
        local_fields_folder=local_fields_folder,
        seed=seed,
    )
    sample_indices = np.arange(len(radius_values), dtype=int)
    data = np.stack(
        [
            mean_strain_x_values,
            mean_strain_y_values,
            mean_strain_xy_values,
            center_points_x_values,
            center_points_y_values,
            radius_values,
            global_mesh_refinement_values,
            hole_mesh_refinement_values,
            sample_indices,
        ]
    ).T
    dataframes = process_map(function, data, max_workers=max_workers)
    return pd.concat(dataframes, ignore_index=True)


def split_train_test(
    data: np.ndarray,  # (Samples, Features)
    test_size: float,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choice n_test_samples elements over iterable members passed as input"""
    total_size = data.shape[0]
    n_test_samples = int(total_size * test_size)

    indices_to_remove = random_generator.choice(
        np.arange(total_size), size=n_test_samples, replace=False
    )
    test_data = data[indices_to_remove]
    train_data = np.delete(data, indices_to_remove, axis=0)
    return train_data, test_data


def main(
    n_samples: int = 1000,
    test_size: float = 0.25,
    seed: int = 69,
    dataset_path: str = "",
    max_workers: int = multiprocessing.cpu_count(),
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Plate parameters
    assert dataset_path, "Must specify dataset path"
    n_test_samples = int(test_size * n_samples)
    n_train_samples = n_samples - n_test_samples
    plate_width_height = 100
    padding_factor = 0.01
    # stress_range = (-1, 1)
    strain_range = (-0.15, 0.15)
    global_mesh_refinement_range = (5, 10)
    hole_mesh_refinement_factor_range = (3, 10)
    # global_mesh_refinement_range = (3, 8)
    # hole_mesh_refinement_factor_range = (8, 15)
    print(f"Dataset folder {dataset_path}")
    print(f"Seed {seed}")
    print(f"Test dataset size {test_size}")
    print(f"Max workers {max_workers}")
    print(f"N train samples {n_train_samples}")
    print(f"N test samples {n_test_samples}")
    print(f"Min strain value {strain_range[0]}")
    print(f"Max strain value {strain_range[1]}")
    print(f"Min global mesh refinement size {global_mesh_refinement_range[0]}")
    print(f"Max global mesh refinement size {global_mesh_refinement_range[1]}")
    print(
        f"Min hole mesh refinement size {hole_mesh_refinement_factor_range[0]}"
    )
    print(
        f"Max hole mesh refinement size {hole_mesh_refinement_factor_range[1]}"
    )

    random_generator = np.random.default_rng(seed=seed)

    (
        total_mean_strain_values_x,
        total_mean_strain_values_y,
        total_mean_strain_values_xy,
        total_center_points_x,
        total_center_points_y,
        total_radius_values,
        total_global_mesh_refinement_values,
        total_hole_mesh_refinement_values,
    ) = _compute_random_dataset_combinations(
        random_generator,
        plate_width_height,
        plate_width_height,
        padding_factor,
        strain_range,
        global_mesh_refinement_range,
        hole_mesh_refinement_factor_range,
        n_samples=n_samples,
    )
    total_data = np.vstack(
        [
            total_mean_strain_values_x,
            total_mean_strain_values_y,
            total_mean_strain_values_xy,
            total_center_points_x,
            total_center_points_y,
            total_radius_values,
            total_global_mesh_refinement_values,
            total_hole_mesh_refinement_values,
        ]
    ).T

    train_data, test_data = split_train_test(
        total_data, test_size, random_generator
    )

    # Main dataset folder
    for dataset_kind, data in zip(("train", "test"), (train_data, test_data)):
        dataset_folder = Path(dataset_path) / Path(dataset_kind)
        dataset_folder.mkdir(parents=True, exist_ok=False)

        (
            mean_strain_values_x,
            mean_strain_values_y,
            mean_strain_values_xy,
            center_points_x,
            center_points_y,
            radius_values,
            global_mesh_refinement_values,
            hole_mesh_refinement_values,
        ) = data.T
        dataframe = generate_and_save_samples(
            mean_strain_x_values=mean_strain_values_x,
            mean_strain_y_values=mean_strain_values_y,
            mean_strain_xy_values=mean_strain_values_xy,
            center_points_x_values=center_points_x,
            center_points_y_values=center_points_y,
            radius_values=radius_values,
            global_mesh_refinement_values=global_mesh_refinement_values,
            hole_mesh_refinement_values=hole_mesh_refinement_values,
            plate_width_height=plate_width_height,
            dataset_folder=dataset_folder,
            seed=seed,
            max_workers=max_workers,
        )
        dataframe.to_csv(
            (dataset_folder / "dataset.csv").as_posix(), index=False
        )


if __name__ == "__main__":
    start_time = time.perf_counter()
    Fire(main)
    perf_time = time.perf_counter() - start_time
    print(f"Data generated in {perf_time:9.4f} seconds")
