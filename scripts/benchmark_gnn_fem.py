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

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import fedoo as fd
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import simcoon
import torch
import torch_geometric as PyG
from fire import Fire
from microgen.remesh import is_periodic

from gnn_local_stress import models
from gnn_local_stress.convert_utils import mesh_to_graph
from gnn_local_stress.datasets import (
    NodeType,
    _compute_node_distances_as_edge_weights,
    compute_node_labels,
    compute_periodic_graph,
)

GPU_ONLY = True


def run_benchmark_fem_task(args: Tuple[fd.Mesh, np.ndarray, bool]) -> float:
    """Wrapper function to run benchmark_fem with independent timing."""
    fd_mesh, mean_strain, hyperelastic = args
    return benchmark_fem(fd_mesh, mean_strain, hyperelastic)


def benchmark_fem_parallel(
    fd_mesh: fd.Mesh,
    strain_samples: List[np.ndarray],
    hyperelastic: bool,
    max_workers: int = 4,
) -> List[float]:
    """Run multiple FEM benchmarks in parallel."""
    tasks = [
        (fd_mesh.copy(), strain, hyperelastic) for strain in strain_samples
    ]

    times: List[float] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_benchmark_fem_task, t): t for t in tasks
        }
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                times.append(result)
            except Exception as e:
                print(f"FEM task failed: {e}")
                times.append(0.0)
    return times


# Benchmarking
@torch.no_grad()
def benchmark_gnn(
    model: models.EncodeProcessDecode,
    pv_mesh: pv.UnstructuredGrid,
    mean_stress: tuple[float, float, float] | np.ndarray,
    device: str,
    use_preprocessing: bool = False,
) -> float:
    torch.cuda.empty_cache()
    if use_preprocessing:
        torch.cuda.synchronize()
        start = time.perf_counter()
    graph = convert_mesh_to_graph(pv_mesh, mean_stress, device)
    if not use_preprocessing:
        torch.cuda.synchronize()
        start = time.perf_counter()
    model.forward(graph)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start


# Benchmarking
def benchmark_fem(
    fd_mesh: fd.Mesh, mean_strain: np.ndarray, hyperelastic: bool
) -> float:
    start = time.perf_counter()
    try:
        if hyperelastic:
            compute_mechanical_fields_hyperelast(fd_mesh, *mean_strain)
        else:
            compute_mechanical_fields(fd_mesh, *mean_strain)
    except Exception as e:
        print(f"An error has occurred {e} of type {type(e)}")
        return 0

    end = time.perf_counter()
    return end - start


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
    shape = shape.extract_surface()
    assert is_periodic(shape.points[:, :-1]), "Mesh is not periodic"
    return shape


def compute_mechanical_fields(
    mesh: fd.Mesh,
    sigma_xx: float,
    sigma_yy: float,
    sigma_xy: float,
    young_modulus: float = 1e5,
    poisson_ratio: float = 0.3,
) -> np.ndarray:
    mesh.reset_interpolation()
    fd.Assembly.delete_memory()
    # --------------- Pre-Treatment --------------------------------------------------------
    space = fd.ModelingSpace("2Dstress")

    type_el = mesh.elm_type
    center = mesh.nearest_node(mesh.bounding_box.center)
    sigma_xx, sigma_yy, sigma_xy = (
        sigma_xx * mesh.bounding_box.volume,
        sigma_yy * mesh.bounding_box.volume,
        sigma_xy * mesh.bounding_box.volume,
    )
    strain_nodes = mesh.add_virtual_nodes(2)

    material = fd.constitutivelaw.ElasticIsotrop(young_modulus, poisson_ratio)

    # Assembly
    wf = fd.weakform.StressEquilibrium(material)

    assemb = fd.Assembly.create(wf, mesh, type_el)

    # Type of problem
    pb = fd.problem.Linear(assemb)

    # Shall add other conditions later on
    bc_periodic = fd.constraint.PeriodicBC(
        [strain_nodes[0], strain_nodes[1], strain_nodes[0]],
        ["DispX", "DispY", "DispY"],
        dim=2,
    )
    pb.bc.add(bc_periodic)

    pb.bc.add("Dirichlet", strain_nodes[1], "DispX", 0)
    pb.bc.add("Dirichlet", center, "Disp", 0, name="center")

    pb.apply_boundary_conditions()

    pb.bc.remove("_Strain")
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[0]],
        "DispX",
        sigma_xx,
        start_value=0,
        name="_Strain",
    )  # EpsXX
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[1]],
        "DispY",
        sigma_yy,
        start_value=0,
        name="_Strain",
    )  # EpsYY
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[0]],
        "DispY",
        sigma_xy,
        start_value=0,
        name="_Strain",
    )  # 2EpsXY

    pb.apply_boundary_conditions()

    pb.solve()

    stress_field_per_node = pb.get_results(assemb, "Stress", "Node")["Stress"]
    mesh.reset_interpolation()
    fd.Assembly.delete_memory()
    return stress_field_per_node[:, :-2]


def compute_mechanical_fields_hyperelast(
    mesh: fd.Mesh,
    eps_xx: float,
    eps_yy: float,
    gamma_xy: float,
    young_modulus: float = 1e5,
    poisson_ratio: float = 0.3,
) -> np.ndarray:
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
    pb.nlsolve(dt=0.02, update_dt=True, interval_output=0.05, print_info=1)
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
    return stress_field_per_node


def convert_mesh_to_graph(
    mesh: pv.UnstructuredGrid,
    mean_stress: tuple[float, float, float],
    device: str,
) -> PyG.data.Data:
    graph = mesh_to_graph(mesh)

    # graph.mean_stress = mean_stress
    graph.edge_attr = _compute_node_distances_as_edge_weights(graph)
    graph = compute_periodic_graph(graph)
    graph.pos = (
        graph.pos[:, :2].float().to(device)
    )  # Remove Z coordinate and cast to float32
    graph.is_periodic = True
    mean_stress = (
        torch.ones((graph.num_nodes, 3)) * torch.Tensor(mean_stress)
    ).to(device)
    graph.mean_stress = mean_stress
    graph.surfaces_nodes_for_div = torch.unsqueeze(
        torch.from_numpy(
            compute_node_labels(
                mesh,
            )
        ),
        1,
    ).to(device)
    graph.nodes_types = torch.clone(graph.surfaces_nodes_for_div)
    return graph.to(device)


def plot_data_benchmark(
    data: dict[str, object],
    output_file: str | None = None,
    hyperelastic: bool | None = True,
) -> None:
    # Set the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Configure font and LaTeX rendering
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlesize": 14,
        }
    )

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the data
    ax.plot(
        data["n_nodes"],
        data["gnn_gpu"],
        label="GNN (GPU)",
        color="C0",
        marker="o",
    )
    ax.plot(
        data["n_nodes"],
        data["gnn_gpu_prepro"],
        label="GNN with Periodic Edges (GPU)",
        color="C1",
        marker="s",
    )
    ax.plot(data["n_nodes"], data["fem"], label="FEM", color="C3", marker="^")

    # Labels and title
    ax.set_xlabel(r"Number of nodes")
    ax.set_ylabel(r"Time [s]")
    ax.set_yscale("log")
    if hyperelastic:
        ax.set_title(
            "Computation Time vs. Number of Nodes (Non linear hyper-elasticity)"
        )
    else:
        ax.set_title("Computation Time vs. Number of Nodes (Linear elasticity)")
    # Grid and legend
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save to file
    if output_file:
        plt.savefig(output_file)
    else:
        # Show the plot
        plt.show()


@torch.no_grad()
def main(csv_data_filename: str = None, hyperelastic: bool = True) -> None:
    if not csv_data_filename:
        seed = 69
        rng = np.random.default_rng(seed=seed)
        global_mesh_refinement_size = 5
        steps = 20
        hole_refinement_factors = np.linspace(1, 100, steps)
        times_gnn_gpu: list[float] = []
        times_gnn_with_preprocessing_gpu: list[float] = []
        times_fem: list[float] = []
        n_nodes: list[int] = []
        input_nodes_features_size = 6
        output_nodes_features_size = 3
        device = "cuda"
        model = models.EncodeProcessDecode(
            input_edges_features_size=1,
            input_nodes_features_size=input_nodes_features_size,
            message_passing_steps=10,
            latent_size=128,
            output_nodes_features_size=output_nodes_features_size,
            mean_mean_stress=torch.Tensor(1).to(device),
            std_mean_stress=torch.Tensor(1).to(device),
            mean_local_stress=torch.Tensor(1).to(device),
            std_local_stress=torch.Tensor(1).to(device),
            mean_pos=torch.Tensor(1).to(device),
            std_pos=torch.Tensor(1).to(device),
            mean_edge_weight=torch.Tensor(1).to(device),
            std_edge_weight=torch.Tensor(1).to(device),
        )
        model = model.to(device)  # type: ignore
        model.eval()
        n_mean_steps = 5
        strain_range = (-0.15, 0.15) if hyperelastic else (-0.05, 0.05)

        for iteration_exec, hole_refinement_factor in enumerate(
            hole_refinement_factors
        ):
            print(f"ITERATION = {iteration_exec}")
            mean_strain = rng.uniform(
                low=strain_range[0],
                high=strain_range[1],
                size=(3),
            )
            pv_mesh = hole_plate_mesh(
                100,
                100,
                30,
                (50, 50),
                hole_refinement_factor,
                global_mesh_refinement_size,
            )
            fd_mesh = fd.Mesh.from_pyvista(pv_mesh).as_2d()
            n_nodes.append(pv_mesh.n_points)
            _ = benchmark_fem(fd_mesh, mean_strain, hyperelastic)
            _ = benchmark_gnn(
                model, pv_mesh, mean_strain, "cuda", use_preprocessing=False
            )  # Dummy launch for not taking into account torch JIT compilation
            # graph = convert_mesh_to_graph(pv_mesh, True, False, mean_strain, device)
            time_gnn_gpu = 0.0
            time_gnn_gpu_prepro = 0.0
            time_fem = 0.0
            fd_mesh = fd.Mesh.from_pyvista(pv_mesh).as_2d()
            strain_samples = [
                rng.uniform(low=strain_range[0], high=strain_range[1], size=3)
                for _ in range(n_mean_steps)
            ]
            fem_times = benchmark_fem_parallel(
                fd_mesh, strain_samples, hyperelastic, max_workers=n_mean_steps
            )
            time_fem = float(np.mean(fem_times))
            for mean_strain in strain_samples:

                time_gnn_gpu += benchmark_gnn(
                    model, pv_mesh, mean_strain, "cuda", use_preprocessing=False
                )
                time_gnn_gpu_prepro += benchmark_gnn(
                    model, pv_mesh, mean_strain, "cuda", use_preprocessing=True
                )

            times_gnn_gpu.append(time_gnn_gpu / n_mean_steps)
            times_gnn_with_preprocessing_gpu.append(
                time_gnn_gpu_prepro / n_mean_steps
            )
            times_fem.append(time_fem)
        data_results = {
            "gnn_gpu": times_gnn_gpu,
            "gnn_gpu_prepro": times_gnn_with_preprocessing_gpu,
            "fem": times_fem,
            "n_nodes": n_nodes,
        }
    else:
        data_results = pd.read_csv(csv_data_filename)
        plot_data_benchmark(data_results, hyperelastic=hyperelastic)
    if hyperelastic:
        plot_filename = "benchmark_fem_hyperelastic_vs_gnn.pdf"
    else:
        plot_filename = "benchmark_fem_vs_gnn.pdf"
    plot_data_benchmark(data_results, plot_filename, hyperelastic=hyperelastic)
    df = pd.DataFrame(data_results)
    csv_filename = "benchmark_data.csv"
    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    Fire(main)
