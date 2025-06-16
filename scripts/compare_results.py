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

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Optional

import fedoo as fd
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

pv.global_theme.font.family = "times"
import scipy
from fire import Fire
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

from gnn_local_stress import convert_utils, data_utils, datasets

# pv.start_xvfb()
WINDOW_SIZE = 2000, 2000
CPU_COUNT = 5
# WINDOW_SIZE = None
SHOW_EDGES = False
USE_GT_RANGE = True
ERROR_ABS = False
KDE = False
PLOT_DATA = False
BINS = 100
# In[3]:


def _concatenate_pdf_unix(pdf_paths: list[str], output_pdf_path: str) -> None:
    command_line = (
        "pdfjam "
        + " ".join(pdf_paths)
        + f" --nup 1x{len(pdf_paths)} --outfile {output_pdf_path}"
    )
    print(command_line)
    os.system(command_line)


def _crop_pdf_to_visible_content_unix(
    input_pdf_path: str, output_pdf_path: str
) -> None:
    os.system(f"pdfcrop {input_pdf_path} {output_pdf_path}")


def _crop_pdf_to_visible_content(
    input_pdf_path: str, output_pdf_path: str, dpi: int = 72
) -> None:
    """
    Crops the input PDF to the minimal bounding box containing all visible content, including text, images, and graphics.

    Parameters:
    - pdf_path (str): Path to the input PDF file.
    - dpi (int): The resolution (dots per inch) to render the PDF page for content detection.

    """

    # Open the PDF
    with fitz.open(input_pdf_path) as document:
        for page in document:
            # Render the page to a pixmap with specified DPI
            pix = page.get_pixmap(dpi=dpi)

            # Convert pixmap to a NumPy array for pixel analysis
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Find rows and columns that are not completely white
            if pix.n == 4:  # RGBA
                mask = (img_array[:, :, :3] < 255).any(
                    axis=2
                )  # Ignore alpha channel
            else:  # RGB
                mask = (img_array < 255).any(axis=2)

            # Get bounding box of non-white content
            coords = np.argwhere(mask)
            if coords.size == 0:
                # Skip empty pages with no visible content
                continue
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)

            # Convert pixel coordinates back to PDF coordinates
            pdf_rect = fitz.Rect(
                x0 / dpi * 72, y0 / dpi * 72, x1 / dpi * 72, y1 / dpi * 72
            )
            bottom_offset = 5
            pdf_rect.y1 += bottom_offset

            # Apply the bounding box as the crop box
            page.set_cropbox(pdf_rect)

        # Save the cropped PDF
        document.save(output_pdf_path)


def compute_divergence_norm_field(
    local_stress_field: np.ndarray,
    op_div_matrix: scipy.sparse.csr,
    surface_nodes_ids: np.ndarray,
) -> np.ndarray:
    stress_x_xy = local_stress_field[:, [0, 2]].T.reshape(-1)
    stress_xy_y = local_stress_field[:, [2, 1]].T.reshape(-1)
    stress_x_xy_xy_y = np.stack([stress_x_xy, stress_xy_y], axis=1)  # 2Nx2
    div_sigma = op_div_matrix @ stress_x_xy_xy_y
    external_boundary_nodes_mask = (
        surface_nodes_ids == datasets.NodeType.EXTERNAL_BOUNDARY
    ).squeeze()
    internal_boundary_nodes_mask = (
        surface_nodes_ids == datasets.NodeType.INTERNAL_BOUNDARY
    ).squeeze()
    div_sigma[external_boundary_nodes_mask] = 0
    # div_sigma[internal_boundary_nodes_mask] = 0

    divergence_field = np.linalg.norm(div_sigma, axis=1)
    return divergence_field


def _plot_fields_row(
    mesh: pv.UnstructuredGrid,
    fields: dict[str, np.ndarray],
    output_file: str,
    clim: Optional[list[tuple[float, float]]] = None,
    show_scalar_bar: bool = True,
) -> None:
    pl = pv.Plotter(
        shape=(1, 3),
        border_width=0.001,
        border_color="white",
        off_screen=not PLOT_DATA,
        window_size=WINDOW_SIZE,
    )
    for i, data_name in enumerate(fields.keys()):
        pl.subplot(0, i)
        mesh.point_data[data_name] = fields[data_name]
        pl.add_mesh(
            mesh.copy(),
            scalars=data_name,
            cmap="jet",
            show_scalar_bar=False,
            clim=clim[i] if (USE_GT_RANGE and clim) else None,
            show_edges=SHOW_EDGES,
            # scalar_bar_args=sargs,
        )
        if show_scalar_bar:
            scalar_bar = pl.add_scalar_bar(
                data_name,
                # label_font_size=int(
                #     pl.window_size[1] / pl.renderers.shape[1] * 0.6 / 22
                # ),
                label_font_size=28,
                bold=True,
                color="Black",
                # bold=True,
                position_x=0.15,
                position_y=0.3,
                n_labels=3,
                width=0.7,
                height=0.03,
                vertical=False,
                fmt="%.2e",
                # n_colors= 10
                # title_font_size=12,
            )
            # scalar_bar.SetTextPositionToPrecedeScalarBar()
            scalar_bar.GetTitleTextProperty().SetBold(True)
            scalar_bar.GetTitleTextProperty().SetFontSize(30)
        # _center_plotter_camera(pl, mesh)
        pl.view_xy()
        # pl.camera.zoom(1.5)
        pl.camera.zoom("tight")
        pl.camera.zoom(0.95)
    if output_file:
        with NamedTemporaryFile(suffix=".pdf") as temp_file:
            pl.save_graphic(temp_file.name)  # , raster=False, painter=False)
            _crop_pdf_to_visible_content(temp_file.name, output_file)
    if PLOT_DATA:
        pl.show()
    pl.close()
    pl.deep_clean()


def plot_difference_baseline_proposed_fem(
    mesh: pv.UnstructuredGrid,
    predicted_local_stress_baseline: np.ndarray,  # Shape (N, 3)
    predicted_local_stress_proposed: np.ndarray,  # Shape (N, 3)
    gt_local_stress: np.ndarray,  # Shape (N, 3)
    baseline_model_name: str,
    proposed_model_name: str,
    output_file: str = "",
) -> None:
    error_baseline_gnn_fem = normalized_mse_loss_element_wise(
        predicted_local_stress=predicted_local_stress_baseline,
        ground_truth_local_stress=gt_local_stress,
    )
    error_proposed_gnn_fem = normalized_mse_loss_element_wise(
        predicted_local_stress=predicted_local_stress_proposed,
        ground_truth_local_stress=gt_local_stress,
    )

    nmse_baseline_gnn_fem_fields: dict[str, np.ndarray] = {}
    nmse_proposed_gnn_fem_fields: dict[str, np.ndarray] = {}
    for model_name, fields, nmse_data in zip(
        [baseline_model_name, proposed_model_name],
        [nmse_baseline_gnn_fem_fields, nmse_proposed_gnn_fem_fields],
        [error_baseline_gnn_fem, error_proposed_gnn_fem],
    ):
        for i, component in enumerate(["XX", "YY", "XY"]):
            tag_name = model_name + " NMSE " + f"Stress {component}"
            fields[tag_name] = nmse_data[:, i]

    with (
        NamedTemporaryFile(suffix=".pdf") as baseline_plot,
        NamedTemporaryFile(suffix=".pdf") as proposed_plot,
    ):
        _plot_fields_row(mesh, nmse_baseline_gnn_fem_fields, baseline_plot.name)
        _plot_fields_row(
            mesh,
            nmse_proposed_gnn_fem_fields,
            proposed_plot.name,
            clim=[
                (error.min(), error.max()) for error in error_baseline_gnn_fem.T
            ],
        )
        _concatenate_pdf_unix(
            [baseline_plot.name, proposed_plot.name], output_file
        )
        _crop_pdf_to_visible_content_unix(output_file, output_file)


def plot_baseline_proposed_fem_divergence_fields(
    mesh: pv.UnstructuredGrid,
    divergence_field_baseline: np.ndarray,  # Shape (N, 3)
    divergence_field_proposed: np.ndarray,  # Shape (N, 3)
    divergence_field_fem: np.ndarray,  # Shape (N, 3)
    baseline_model_name: str,
    proposed_model_name: str,
    output_file: str = "",
) -> None:
    fields = {
        f"{baseline_model_name} Divergence Field": divergence_field_baseline,
        f"{proposed_model_name} Divergence Field": divergence_field_proposed,
        "FEM Divergence Field": divergence_field_fem,
    }
    fem_min_max = (
        fields["FEM Divergence Field"].min(),
        fields["FEM Divergence Field"].max(),
    )

    _plot_fields_row(mesh, fields, output_file, clim=[fem_min_max] * 3)


def plot_baseline_proposed_fem(
    mesh: pv.UnstructuredGrid,
    predicted_local_stress_baseline: np.ndarray,  # Shape (N, 3)
    predicted_local_stress_proposed: np.ndarray,  # Shape (N, 3)
    gt_local_stress: np.ndarray,  # Shape (N, 3)
    baseline_model_name: str,
    proposed_model_name: str,
    output_file: str = "",
) -> None:

    baseline_gnn_fields: dict[str, np.ndarray] = {}
    proposed_gnn_fields: dict[str, np.ndarray] = {}
    fem_fields: dict[str, np.ndarray] = {}

    for model_name, fields, field_data in zip(
        [baseline_model_name, proposed_model_name, "FEM"],
        [baseline_gnn_fields, proposed_gnn_fields, fem_fields],
        [
            predicted_local_stress_baseline,
            predicted_local_stress_proposed,
            gt_local_stress,
        ],
    ):
        for i, component in enumerate(["XX", "YY", "XY"]):
            tag_name = model_name + f" Stress {component}"
            fields[tag_name] = field_data[:, i]

    fem_clim = [(data.min(), data.max()) for data in gt_local_stress.T]
    with (
        NamedTemporaryFile(suffix=".pdf") as baseline_plot,
        NamedTemporaryFile(suffix=".pdf") as proposed_gnn_plot,
        NamedTemporaryFile(suffix=".pdf") as fem_plot,
    ):
        _plot_fields_row(
            mesh,
            baseline_gnn_fields,
            baseline_plot.name,
            fem_clim,
            show_scalar_bar=True,
        )
        _plot_fields_row(
            mesh,
            proposed_gnn_fields,
            proposed_gnn_plot.name,
            clim=fem_clim,
            show_scalar_bar=True,
        )
        _plot_fields_row(mesh, fem_fields, fem_plot.name, show_scalar_bar=True)
        _concatenate_pdf_unix(
            [baseline_plot.name, proposed_gnn_plot.name, fem_plot.name],
            output_file,
        )
        _crop_pdf_to_visible_content_unix(output_file, output_file)


def normalized_mse_loss_single(
    ground_truth_local_stress: np.ndarray,
    predicted_local_stress: np.ndarray,
    reduce: bool = True,
) -> float | tuple[float, float, float]:
    mean_gt = ground_truth_local_stress.mean(axis=0)  # Shape == (1,3)
    mse = ((ground_truth_local_stress - predicted_local_stress) ** 2).sum(
        axis=0
    )  # Shape == (3)
    normalization_term = ((ground_truth_local_stress - mean_gt) ** 2).sum(
        axis=0
    )  # .sum(axis=0)  # Shape == (3)
    loss = mse / normalization_term
    if reduce:
        return loss.mean()  # Shape (1)
    return loss


def normalized_mse_loss_element_wise(
    ground_truth_local_stress: np.ndarray,
    predicted_local_stress: np.ndarray,
    reduce: bool = True,
) -> float | tuple[float, float, float]:
    mean_gt = ground_truth_local_stress.mean(axis=0)  # Shape == (1,3)
    mse = (
        ground_truth_local_stress - predicted_local_stress
    ) ** 2  # Shape == (N, 3)
    normalization_term = (((ground_truth_local_stress - mean_gt) ** 2)).sum(
        axis=0
    )  # Shape == (1, 3)
    loss = mse / normalization_term
    return loss


def plot_histogram(
    data: np.ndarray,
    bins: int = 10,
    title: str = "Histogram",
    xlabel: str = "Values",
    ylabel: str = "Frequency",
    output_file: str = "",
    show_plot: bool = False,
):
    """
    Plots a publication-quality histogram from a 1D NumPy array.

    Parameters:
    - data (np.ndarray): 1D array containing data points
    - bins (int): Number of bins for the histogram
    - title (str): Title of the histogram plot
    - xlabel (str): Label for the x-axis
    - ylabel (str): Label for the y-axis
    - output_file (str): Path to save the figure (if provided)
    - show_plot (bool): Whether to display the plot
    """

    plt.figure(figsize=(8, 6), dpi=300)  # High-res figure
    plt.hist(data, bins=bins, edgecolor="black", linewidth=1.2)

    # Use bold, readable font for labels and title
    font_kwargs = {"fontsize": 14, "fontweight": "bold", "family": "serif"}
    plt.title(title, **font_kwargs)
    plt.xlabel(xlabel, **font_kwargs)
    plt.ylabel(ylabel, **font_kwargs)

    # Customize ticks
    plt.xticks(fontsize=12, fontweight="bold", family="serif")
    plt.yticks(fontsize=12, fontweight="bold", family="serif")

    # Clean layout
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()


def plot_histogram_old(
    data: np.ndarray,
    bins: int = 10,
    title: str = "Histogram",
    xlabel: str = "Values",
    ylabel: str = "Frequency",
    output_file: str = "",
):
    """
    Plots a histogram from a 1D NumPy array.

    Parameters:
    - data (np.ndarray): 1D array containing data points
    - bins (int): Number of bins for the histogram (default is 10)
    - title (str): Title of the histogram plot (default is 'Histogram')
    - xlabel (str): Label for the x-axis (default is 'Values')
    - ylabel (str): Label for the y-axis (default is 'Frequency')
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, edgecolor="black")

    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Display the plot
    if output_file:
        plt.savefig(output_file)
    if PLOT_DATA:
        plt.show()


def plot_two_histograms(
    data1: np.ndarray,
    data2: np.ndarray,
    bins: int = 50,
    title: str = "Histograms",
    xlabel: str = "Values",
    ylabel: str = "Frequency",
    label1: str = "Data 1",
    label2: str = "Data 2",
    alpha1: float = 0.5,
    alpha2: float = 0.5,
    kde: bool = False,
    output_file: str = "",
    show_plot: bool = False,
) -> None:
    """
    Plots two histograms from two 1D NumPy arrays with optional KDE curves for comparison.

    Parameters:
    - data1, data2: 1D arrays for the datasets
    - bins: Number of bins for the histograms
    - title, xlabel, ylabel: Title and axis labels
    - label1, label2: Labels for the histograms
    - alpha1, alpha2: Transparency for each histogram
    - kde: Whether to overlay KDE curves
    - output_file: File path to save the figure
    - show_plot: Whether to display the plot
    """
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot histograms
    plt.hist(
        data1,
        bins=bins,
        alpha=alpha1,
        label=label1,
        edgecolor="black",
        density=True,
        linewidth=1.2,
    )
    plt.hist(
        data2,
        bins=bins,
        alpha=alpha2,
        label=label2,
        edgecolor="black",
        density=True,
        linewidth=1.2,
    )

    # Optional KDE overlays
    if kde:
        kde1 = gaussian_kde(data1)
        x1 = np.linspace(min(data1), max(data1), 1000)
        plt.plot(
            x1,
            kde1(x1),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"{label1} KDE",
        )

        kde2 = gaussian_kde(data2)
        x2 = np.linspace(min(data2), max(data2), 1000)
        plt.plot(
            x2,
            kde2(x2),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"{label2} KDE",
        )

    # Styling for publication
    font_kwargs = {"fontsize": 14, "fontweight": "bold", "family": "serif"}
    plt.title(title, **font_kwargs)
    plt.xlabel(xlabel, **font_kwargs)
    plt.ylabel(ylabel, **font_kwargs)

    plt.xticks(fontsize=12, fontweight="bold", family="serif")
    plt.yticks(fontsize=12, fontweight="bold", family="serif")
    plt.legend(prop={"size": 12, "weight": "bold", "family": "serif"})

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
    elif show_plot:
        plt.show()


def plot_two_histograms_old(
    data1: np.ndarray,
    data2: np.ndarray,
    bins: int = 50,
    title: str = "Histograms",
    xlabel: str = "Values",
    ylabel: str = "Frequency",
    label1: str = "Data 1",
    label2: str = "Data 2",
    alpha1: float = 0.5,
    alpha2: float = 0.5,
    kde: bool = False,
    output_file: str = "",
) -> None:
    """
    Plots two histograms from two 1D NumPy arrays with optional KDE curves for comparison.

    Parameters:
    - data1 (np.ndarray): 1D array for the first dataset
    - data2 (np.ndarray): 1D array for the second dataset
    - bins (int): Number of bins for the histograms (default is 10)
    - title (str): Title of the histogram plot (default is 'Histograms')
    - xlabel (str): Label for the x-axis (default is 'Values')
    - ylabel (str): Label for the y-axis (default is 'Frequency')
    - label1 (str): Label for the first dataset (default is 'Data 1')
    - label2 (str): Label for the second dataset (default is 'Data 2')
    - alpha (float): Transparency level for the histograms (default is 0.5, range is 0 to 1)
    - kde (bool): Whether to plot the KDE curve for each dataset (default is True)
    """
    plt.figure(figsize=(8, 6))

    # Plot histogram for the first dataset
    plt.hist(
        data1,
        bins=bins,
        alpha=alpha1,
        label=label1,
        edgecolor="black",
        density=True,
    )

    # Plot histogram for the second dataset
    plt.hist(
        data2,
        bins=bins,
        alpha=alpha2,
        label=label2,
        edgecolor="black",
        density=True,
    )

    # Add KDE curves if requested
    if kde:
        # KDE for the first dataset
        kde1 = gaussian_kde(data1)
        x1 = np.linspace(min(data1), max(data1), 1000)
        plt.plot(
            x1,
            kde1(x1),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"{label1} KDE",
        )

        # KDE for the second dataset
        kde2 = gaussian_kde(data2)
        x2 = np.linspace(min(data2), max(data2), 1000)
        plt.plot(
            x2,
            kde2(x2),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"{label2} KDE",
        )

    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add a legend to distinguish between the datasets
    plt.legend()

    # Display the plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    if PLOT_DATA:
        plt.show()


def init_op_div_matrix(
    mesh_data: dict[str, np.ndarray], hyperelastic_div: bool = False
) -> scipy.sparse.csr:
    n_nodes = mesh_data["stress_field"].shape[0]

    data = mesh_data["op_div_matrix_data"].astype(float)
    col_indices = mesh_data["op_div_matrix_col_indices"].astype(int)
    row_indices = mesh_data["op_div_matrix_row_indices"].astype(int)
    shape = tuple(mesh_data["op_div_matrix_shape"].astype(int))
    # Create a torch sparse tensor
    # FIX TO DEAL WITH VIRTUAL NODES
    op_div_matrix = scipy.sparse.coo_matrix(
        (data, (row_indices, col_indices)), shape=shape
    )
    return op_div_matrix.tocsr()


def compute_divergence(  # compute_pinn_loss
    local_stress_field: np.ndarray,
    op_div_matrix: scipy.sparse.csr,
    surface_nodes_ids: np.ndarray,
    reduce_strategy: str = "square",
) -> np.ndarray:
    if reduce_strategy not in ("abs", "square"):
        raise AttributeError("reduce_strategy must be 'abs' or 'square'")
    stress_x_xy = local_stress_field[:, [0, 2]].T.reshape(-1)
    stress_xy_y = local_stress_field[:, [2, 1]].T.reshape(-1)
    stress_x_xy_xy_y = np.stack([stress_x_xy, stress_xy_y], axis=1)  # 2Nx2
    div_sigma = op_div_matrix @ stress_x_xy_xy_y
    # div_sigma.shape == (N,2)
    external_boundary_nodes_mask = (
        surface_nodes_ids == datasets.NodeType.EXTERNAL_BOUNDARY
    ).squeeze()
    internal_boundary_nodes_mask = (
        surface_nodes_ids == datasets.NodeType.INTERNAL_BOUNDARY
    ).squeeze()
    div_sigma[external_boundary_nodes_mask] = 0
    div_sigma[internal_boundary_nodes_mask] = 0
    if reduce_strategy == "abs":
        div_sigma = np.abs(div_sigma)
    elif reduce_strategy == "square":
        div_sigma = div_sigma**2
    div_sigma = np.mean(div_sigma, axis=0)  # div_sigma.shape == 2
    return np.sum(div_sigma)  # Tensor[float] Scalar


def compute_divergence_old(
    local_stress_field: np.ndarray,
    op_div_matrix: scipy.sparse.csr,
    surface_nodes_ids: np.ndarray,
    reduce_strategy: str = "square",
) -> np.ndarray:
    if reduce_strategy not in ("abs", "square", "raw"):
        raise AttributeError("reduce_strategy must be 'abs' or 'square'")
    stress_x_xy = local_stress_field[:, [0, 2]].T.reshape(-1)
    stress_xy_y = local_stress_field[:, [2, 1]].T.reshape(-1)
    stress_x_xy_xy_y = np.stack([stress_x_xy, stress_xy_y], axis=1)  # 2Nx2
    div_sigma = op_div_matrix @ stress_x_xy_xy_y
    # div_sigma.shape == (N,2)
    external_boundary_nodes_mask = (
        surface_nodes_ids == datasets.NodeType.EXTERNAL_BOUNDARY
    ).squeeze()
    internal_boundary_nodes_mask = (
        surface_nodes_ids == datasets.NodeType.INTERNAL_BOUNDARY
    ).squeeze()
    div_sigma[external_boundary_nodes_mask] = 0
    div_sigma[internal_boundary_nodes_mask] = 0
    if reduce_strategy == "abs":
        div_sigma = np.abs(div_sigma)
    elif reduce_strategy == "square":
        div_sigma = div_sigma**2
    div_sigma = np.mean(div_sigma, axis=0)  # div_sigma.shape == 2
    return np.sum(div_sigma)  # Tensor[float] Scalar


def topk_indices(arr: np.ndarray, k: int, order: str) -> np.ndarray:
    if order == "high":
        return np.argsort(arr)[-k:][::-1]  # top k highest values indices
    if order == "low":
        return np.argsort(arr)[:k]  # top k lowest values indices
    raise ValueError("Order must be either 'high' or 'low'.")


def von_mises_stress(
    mean_stress_x: np.ndarray,
    mean_stress_y: np.ndarray,
    mean_stress_xy: np.ndarray,
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


def plot_graph_mesh_stress_predicted_and_fem(
    mesh: pv.UnstructuredGrid,
    predicted_local_stress: np.ndarray,  # Shape (N, 3)
    gt_local_stress: np.ndarray,  # Shape (N, 3)
    index_in_dataset: int,
    mean_stress: tuple[float, float, float],
    show_edges: bool = True,
    use_gt_range: bool = True,
    # error_log: bool = False,
    # error_abs: bool = ERROR_ABS,
    model_name: str = "GNN",
    output_file: str = "",
    plot_sample_index: bool = False,
) -> None:
    predicted_stress_field_formatted = (
        convert_utils.format_stress_field_to_fedoo(predicted_local_stress)
    )
    fem_stress_field_formatted = convert_utils.format_stress_field_to_fedoo(
        gt_local_stress
    )
    mean_von_mises = von_mises_stress(*mean_stress)
    pred_local_von_mises = von_mises_stress(*predicted_local_stress.T)
    gt_local_von_mises = von_mises_stress(*gt_local_stress.T)
    error_sample = list(
        normalized_mse_loss_single(
            predicted_local_stress=predicted_local_stress,
            ground_truth_local_stress=gt_local_stress,
            reduce=False,
        )
    )
    error_vm = normalized_mse_loss_single(
        pred_local_von_mises, gt_local_von_mises
    )
    error_sample.append(error_vm)
    """
    if not error_abs:
        error_field = normalized_mse_loss_element_wise(np.concatenate((predicted_local_stress,pred_local_von_mises[:,np.newaxis]), axis=1), 
                                                       np.concatenate((gt_local_stress,gt_local_von_mises[:,np.newaxis]), axis=1))
        error_field = np.log(error_field) if error_log else error_field
    else:
        error_field = np.abs(np.concatenate((predicted_local_stress,pred_local_von_mises[:,np.newaxis]), axis=1) - 
                                                       np.abs(np.concatenate((gt_local_stress,gt_local_von_mises[:,np.newaxis]), axis=1)))
    """
    error_field = normalized_mse_loss_element_wise(
        np.concatenate(
            (predicted_local_stress, pred_local_von_mises[:, np.newaxis]),
            axis=1,
        ),
        np.concatenate(
            (gt_local_stress, gt_local_von_mises[:, np.newaxis]), axis=1
        ),
    )
    dataset = fd.DataSet(fd.Mesh.from_pyvista(mesh))
    dataname_predicted = "Predicted_Stress"
    dataname_fedoo = "GT_Stress"
    dataname_error = "Error_field"
    dataset.node_data[dataname_predicted] = predicted_stress_field_formatted
    dataset.node_data[dataname_fedoo] = fem_stress_field_formatted
    dataset.node_data[dataname_error] = error_field.T
    dataset.mesh.nodes = dataset.mesh.nodes[:, :2]  # To fix fedoo plot
    title_font_size = 20
    scalar_bar_font_size = 16
    scalar_bar_label_font_size = 16
    font_size = 8
    title_size = 12
    pl = pv.Plotter(
        shape=(3, 4), window_size=WINDOW_SIZE, off_screen=not PLOT_DATA
    )
    for i, component in enumerate(["XX", "YY", "XY", "vm"]):
        title_component = component if component != "vm" else "Von Mises"
        pl.subplot(0, i)
        # Show MSE between predicted and fedoo in plot, must be normalized
        if plot_sample_index:
            pl.add_text(
                f"Sample N {index_in_dataset}",
                position="upper_right",
                color="blue",
                shadow=True,
                font_size=font_size,
            )
        fedoo_stress = dataset.get_data(dataname_fedoo, component)
        dataset.plot(
            dataname_predicted,
            component=component,
            plotter=pl,
            title_size=title_size,
            title=f"{model_name} Stress {title_component}",
            clim=(
                [fedoo_stress.min(), fedoo_stress.max()]
                if use_gt_range
                else None
            ),
            show_edges=show_edges,
        )
        pl.subplot(1, i)
        if component == "vm":
            pl.add_text(
                f"Von Mises mean value = {mean_von_mises}",
                position="lower_left",
                color="blue",
                shadow=True,
                font_size=font_size,
            )
        if component != "vm":
            pl.add_text(
                f"Mean Stress {title_component} = {mean_stress[i]}",
                position="lower_left",
                color="blue",
                shadow=True,
                font_size=font_size,
            )
        dataset.plot(
            dataname_fedoo,
            component=component,
            plotter=pl,
            show_edges=show_edges,
            title_size=title_size,
            title=f"FEM Stress {title_component}",
            # clim=[stress_min_max[0], stress_min_max[1]],
        )
        pl.subplot(2, i)
        pl.add_text(
            f"NMSE = {error_sample[i]}",
            position="lower_left",
            color="blue",
            shadow=True,
            font_size=font_size,
        )
        dataset.plot(
            dataname_error,
            component=i,
            plotter=pl,
            show_edges=show_edges,
            title_size=title_size,
            title=f"NMSE {title_component}",
            # clim=[stress_min_max[0], stress_min_max[1]],
        )
    if output_file:
        with NamedTemporaryFile(suffix=".pdf") as temp_file:
            pl.save_graphic(temp_file.name)  # , raster=False, painter=False)
            _crop_pdf_to_visible_content_unix(temp_file.name, output_file)
    if PLOT_DATA:
        pl.show()
    pl.close()
    pl.deep_clean()


# In[4]:


def load_stress_field(dataframe: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.load(filename)["stress_field"]
        for filename in dataframe["data_filename"]
    ]


def load_meshes(dataframe: pd.DataFrame) -> list[pv.UnstructuredGrid]:
    return [
        pv.get_reader(filename).read()
        for filename in dataframe["mesh_filename"]
    ]


def load_divergence_operators(
    dataframe: pd.DataFrame, hyperelastic_div: bool
) -> list[scipy.sparse.csr]:
    return [
        init_op_div_matrix(np.load(filename), hyperelastic_div)
        for filename in dataframe["data_filename"]
    ]


def load_node_labels(
    dataframe: pd.DataFrame,
) -> list[np.ndarray]:
    return [
        np.load(filename)["node_labels"].astype(int)
        for filename in dataframe["data_filename"]
    ]


def load_mean_std_from_json(
    normalization_json_path: str,
) -> tuple[float, float]:

    with open(normalization_json_path) as f:
        data_json = json.load(f)

    mean_local_stress = np.array(data_json["mean_local_stress"])
    std_local_stress = np.array(data_json["std_local_stress"])
    return mean_local_stress, std_local_stress


def _plot_sample_comparisons(
    root_folder_path: Path,
    meshes: list[pv.UnstructuredGrid],
    dataset_results: dict[str, dict[str, np.ndarray]],
    sample_indices: Iterable[int],
    baseline_model_name: str,
    proposed_model_name: str,
) -> None:
    nmse_folder = root_folder_path / "nmse"
    divergence_fields_folder = root_folder_path / "divergence_fields"
    stress_fields_folder = root_folder_path / "stress_fields"
    distributions_folder = root_folder_path / "distributions"
    for folder in (
        nmse_folder,
        divergence_fields_folder,
        stress_fields_folder,
        distributions_folder,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    ###
    def _plot_samples_parallel(
        sample_index_and_topk_index: tuple[int, int],
    ) -> None:
        index, i = sample_index_and_topk_index
        filename_for_plot = (
            stress_fields_folder / f"topk{i+1}_sample_{index}.pdf"
        ).as_posix()
        plot_baseline_proposed_fem(
            mesh=meshes[index],
            predicted_local_stress_baseline=dataset_results[
                baseline_model_name
            ]["local_stress_fields"][index],
            predicted_local_stress_proposed=dataset_results[
                proposed_model_name
            ]["local_stress_fields"][index],
            gt_local_stress=dataset_results["FEM"]["local_stress_fields"][
                index
            ],
            output_file=filename_for_plot,
            baseline_model_name=baseline_model_name,
            proposed_model_name=proposed_model_name,
        )
        filename_for_plot = (
            divergence_fields_folder / f"topk{i+1}_sample_{index}.pdf"
        ).as_posix()
        plot_baseline_proposed_fem_divergence_fields(
            mesh=meshes[index],
            divergence_field_baseline=dataset_results[baseline_model_name][
                "divergence_fields_standard"
            ][index],
            divergence_field_proposed=dataset_results[proposed_model_name][
                "divergence_fields_standard"
            ][index],
            divergence_field_fem=dataset_results["FEM"][
                "divergence_fields_standard"
            ][index],
            output_file=filename_for_plot,
            baseline_model_name=baseline_model_name,
            proposed_model_name=proposed_model_name,
        )
        filename_for_plot = (
            nmse_folder / f"topk{i+1}_sample_{index}.pdf"
        ).as_posix()
        plot_difference_baseline_proposed_fem(
            mesh=meshes[index],
            predicted_local_stress_baseline=dataset_results[
                baseline_model_name
            ]["local_stress_fields"][index],
            predicted_local_stress_proposed=dataset_results[
                proposed_model_name
            ]["local_stress_fields"][index],
            gt_local_stress=dataset_results["FEM"]["local_stress_fields"][
                index
            ],
            output_file=filename_for_plot,
            baseline_model_name=baseline_model_name,
            proposed_model_name=proposed_model_name,
        )
        for j, stress_component in enumerate(("XX", "YY", "XY")):
            distribution_filename = (
                distributions_folder
                / f"topk{i+1}_sample_{index}_distribution_{stress_component}_{proposed_model_name}.pdf"
            ).as_posix()
            plot_two_histograms(
                data1=dataset_results["FEM"]["local_stress_fields_standard"][
                    index
                ][:, j],
                data2=dataset_results[proposed_model_name][
                    "local_stress_fields_standard"
                ][index][:, j],
                bins=BINS,
                label1="FEM",
                label2=proposed_model_name,
                alpha1=1,
                alpha2=0.5,
                title=f"Distribution Stress {stress_component}",
                ylabel="Density",
                kde=KDE,
                output_file=distribution_filename,
            )
            distribution_filename = (
                distributions_folder
                / f"topk{i+1}_sample_{index}_distribution_{stress_component}_{baseline_model_name}.pdf"
            ).as_posix()
            plot_two_histograms(
                data1=dataset_results["FEM"]["local_stress_fields_standard"][
                    index
                ][:, j],
                data2=dataset_results[baseline_model_name][
                    "local_stress_fields_standard"
                ][index][:, j],
                bins=BINS,
                label1="FEM",
                label2=baseline_model_name,
                alpha1=1,
                alpha2=0.5,
                title=f"Distribution Stress {stress_component}",
                ylabel="Density",
                kde=KDE,
                output_file=distribution_filename,
            )

    # threads = multiprocess.cpu_count()
    arguments = [
        (sample_index, topk_index)
        for topk_index, sample_index in enumerate(sample_indices)
    ]
    for args in arguments:
        _plot_samples_parallel(args)
    # with multiprocess.Pool(CPU_COUNT) as pool:
    #     _ = pool.map(_plot_samples_parallel, arguments)


def main(
    figures_folder: str,
    ground_truth_csv: str,
    baseline_model_folder: str,
    proposed_model_folder: str,
    topk: int = 100,
    single_index: int = None,
    hyperelastic_div: bool = False,
) -> None:

    baseline_model_name = baseline_model_folder.split("/")[-1]
    proposed_model_name = proposed_model_folder.split("/")[-1]

    # inferences_folder = Path("inferences/divergence/")
    gt_dataframe: pd.DataFrame = pd.read_csv(ground_truth_csv)
    # Load mesh geometries for plotting
    meshes: list[pv.UnstructuredGrid] = load_meshes(gt_dataframe)
    # Load FEM stress fields
    gt_stress_fields: list[np.ndarray] = load_stress_field(gt_dataframe)

    baseline_model_path = Path(baseline_model_folder)
    mean_local_stress, std_local_stress = load_mean_std_from_json(
        (baseline_model_path / "normalize_params.json").as_posix()
    )
    gt_stress_fields_standardized: list[np.ndarray] = [
        data_utils.standardize(data, mean_local_stress, std_local_stress)
        for data in gt_stress_fields
    ]
    # Load operators for computing divergence
    div_ops: list[scipy.sparse.csr] = load_divergence_operators(
        gt_dataframe, hyperelastic_div
    )

    # Load labels of nodes
    node_labels_per_graph: list[np.ndarray] = load_node_labels(gt_dataframe)

    dataset_results = {
        baseline_model_name: {},
        proposed_model_name: {},
        "FEM": {},
    }
    # Compute divergence scalars for FEM solutions

    dataset_results["FEM"]["divergences_scalars"] = np.array(
        [
            compute_divergence(local_stress_field, op_div, node_labels)
            for local_stress_field, op_div, node_labels in zip(
                gt_stress_fields, div_ops, node_labels_per_graph
            )
        ]
    )
    dataset_results["FEM"]["divergences_scalars_standard"] = np.array(
        [
            compute_divergence(local_stress_field, op_div, node_labels)
            for local_stress_field, op_div, node_labels in zip(
                gt_stress_fields_standardized,
                div_ops,
                node_labels_per_graph,
            )
        ]
    )

    dataset_results["FEM"]["divergence_fields"] = [
        compute_divergence_norm_field(local_stress_field, op_div, node_labels)
        for local_stress_field, op_div, node_labels in zip(
            gt_stress_fields,
            div_ops,
            node_labels_per_graph,
        )
    ]

    dataset_results["FEM"]["divergence_fields_standard"] = [
        compute_divergence_norm_field(local_stress_field, op_div, node_labels)
        for local_stress_field, op_div, node_labels in zip(
            gt_stress_fields_standardized,
            div_ops,
            node_labels_per_graph,
        )
    ]

    dataset_results["FEM"]["local_stress_fields"] = gt_stress_fields
    dataset_results["FEM"][
        "local_stress_fields_standard"
    ] = gt_stress_fields_standardized

    for inferences_folder in (baseline_model_folder, proposed_model_folder):
        model_name = inferences_folder.split("/")[-1]
        inferences_folder = Path(inferences_folder)
        inferences_csv = (inferences_folder / "dataset.csv").as_posix()
        inferences_dataframe = pd.read_csv(inferences_csv)

        pred_stress_fields: list[np.ndarray] = load_stress_field(
            inferences_dataframe
        )

        pred_stress_fields_standardized: list[np.ndarray] = [
            data_utils.standardize(data, mean_local_stress, std_local_stress)
            for data in pred_stress_fields
        ]

        dataset_results[model_name]["local_stress_fields"] = pred_stress_fields
        dataset_results[model_name][
            "local_stress_fields_standard"
        ] = pred_stress_fields_standardized

        mean_stresses = np.array(
            gt_dataframe[["mean_stress_x", "mean_stress_y", "mean_stress_xy"]]
        )

        losses = np.array(
            [
                normalized_mse_loss_single(
                    ground_truth_local_stress=gt, predicted_local_stress=pred
                )
                for pred, gt in zip(gt_stress_fields, pred_stress_fields)
            ]
        )
        losses_standard = np.array(
            [
                normalized_mse_loss_single(
                    ground_truth_local_stress=gt, predicted_local_stress=pred
                )
                for pred, gt in zip(
                    gt_stress_fields_standardized,
                    pred_stress_fields_standardized,
                )
            ]
        )
        dataset_results[model_name]["losses"] = losses
        dataset_results[model_name]["losses_standard"] = losses_standard

        r2_score_raw = np.array(
            [
                r2_score(y_true=gt, y_pred=pred)
                for pred, gt in zip(gt_stress_fields, pred_stress_fields)
            ]
        )
        r2_score_standard = np.array(
            [
                r2_score(y_true=gt, y_pred=pred)
                for pred, gt in zip(
                    gt_stress_fields_standardized,
                    pred_stress_fields_standardized,
                )
            ]
        )
        dataset_results[model_name]["r2_score_raw"] = r2_score_raw
        dataset_results[model_name]["r2_score_standard"] = r2_score_standard

        dataset_results[model_name]["divergences_scalars"] = np.array(
            [
                compute_divergence(local_stress_field, op_div, node_labels)
                for local_stress_field, op_div, node_labels in zip(
                    pred_stress_fields, div_ops, node_labels_per_graph
                )
            ]
        )

        dataset_results[model_name]["divergences_scalars_standard"] = np.array(
            [
                compute_divergence(local_stress_field, op_div, node_labels)
                for local_stress_field, op_div, node_labels in zip(
                    pred_stress_fields_standardized,
                    div_ops,
                    node_labels_per_graph,
                )
            ]
        )

        dataset_results[model_name]["divergence_fields"] = [
            compute_divergence_norm_field(
                local_stress_field, op_div, node_labels
            )
            for local_stress_field, op_div, node_labels in zip(
                pred_stress_fields,
                div_ops,
                node_labels_per_graph,
            )
        ]
        dataset_results[model_name]["divergence_fields_standard"] = [
            compute_divergence_norm_field(
                local_stress_field, op_div, node_labels
            )
            for local_stress_field, op_div, node_labels in zip(
                pred_stress_fields_standardized,
                div_ops,
                node_labels_per_graph,
            )
        ]
    # PLOTTING DATASET DISTRIBUTIONS

    figures_folder_path = Path(figures_folder)
    figures_folder_path.mkdir(parents=True, exist_ok=True)

    for model_name in (baseline_model_name, proposed_model_name):
        model_folder = figures_folder_path / model_name
        model_folder.mkdir(parents=True, exist_ok=True)
        distributions_folder = model_folder / "distributions"
        distributions_folder.mkdir(parents=True, exist_ok=True)
        filename_for_plot = (
            distributions_folder / "divergence_loss_distribution_raw.pdf"
        ).as_posix()
        plot_two_histograms(
            data1=dataset_results["FEM"]["divergences_scalars"],
            data2=dataset_results[model_name]["divergences_scalars"],
            title="Divergence loss distribution",
            bins=BINS,
            xlabel="Divergence loss values",
            ylabel="Density",
            label1="FEM",
            label2=model_name,
            alpha1=1,
            alpha2=0.5,
            kde=KDE,
            output_file=filename_for_plot,
        )
        filename_for_plot = (
            distributions_folder / "divergence_loss_distribution_standard.pdf"
        ).as_posix()
        plot_two_histograms(
            data1=dataset_results["FEM"]["divergences_scalars_standard"],
            data2=dataset_results[model_name]["divergences_scalars_standard"],
            bins=BINS,
            label1="FEM",
            title="Divergence loss distribution",
            xlabel="Divergence loss values",
            ylabel="Density",
            label2=model_name,
            alpha1=1,
            alpha2=0.5,
            kde=KDE,
            output_file=filename_for_plot,
        )

        filename_for_plot = (
            distributions_folder / "nmse_distribution.pdf"
        ).as_posix()
        plot_histogram(
            dataset_results[model_name]["losses"],
            bins=BINS,
            title="NMSE distribution",
            xlabel="NMSE values",
            output_file=filename_for_plot,
        )
        for i, stress_component in enumerate(("XX", "YY", "XY")):
            filename_for_plot = (
                distributions_folder / f"distribution_{stress_component}.pdf"
            ).as_posix()
            plot_two_histograms(
                data1=np.concatenate(
                    dataset_results["FEM"]["local_stress_fields_standard"],
                    axis=0,
                )[:, i],
                data2=np.concatenate(
                    dataset_results[model_name]["local_stress_fields_standard"],
                    axis=0,
                )[:, i],
                bins=BINS,
                label1="FEM",
                label2=model_name,
                alpha1=1,
                alpha2=0.5,
                title=f"Distribution Stress {stress_component}",
                ylabel="Density",
                kde=KDE,
                output_file=filename_for_plot,
            )
    ### Filter results where Div(GNN(M)) < Div(FEM(M))
    print(
        f"{baseline_model_name} Mean Loss = {np.mean(dataset_results[baseline_model_name]['losses_standard'])}"
    )
    print(
        f"{proposed_model_name} Mean Loss = {np.mean(dataset_results[proposed_model_name]['losses_standard'])}"
    )
    print(
        f"{baseline_model_name} Mean Divergence = {np.mean(dataset_results[baseline_model_name]['divergences_scalars_standard'])}"
    )
    print(
        f"{proposed_model_name} Mean Divergence = {np.mean(dataset_results[proposed_model_name]['divergences_scalars_standard'])}"
    )
    print(
        f"FEM Mean Divergence = {np.mean(dataset_results['FEM']['divergences_scalars_standard'])}"
    )

    if single_index:  # Maybe change this in future for plotting several indices
        single_index_folder = figures_folder_path / "single_index"
        single_index_folder.mkdir(parents=True, exist_ok=True)
        _plot_sample_comparisons(
            root_folder_path=single_index_folder,
            meshes=meshes,
            dataset_results=dataset_results,
            sample_indices=[single_index],
            baseline_model_name=baseline_model_name,
            proposed_model_name=proposed_model_name,
        )
        print(
            f"{baseline_model_name} Mean Loss of index {single_index} =  {dataset_results[baseline_model_name]['losses'][single_index]}"
        )
        print(
            f"{proposed_model_name} Mean Loss of index {single_index} = {dataset_results[proposed_model_name]['losses'][single_index]}"
        )
        print(
            f"{baseline_model_name} Mean Divergence of index {single_index} = {dataset_results[baseline_model_name]['divergences_scalars_standard'][single_index]}"
        )
        print(
            f"{proposed_model_name} Mean Divergence of index {single_index} =  {dataset_results[proposed_model_name]['divergences_scalars_standard'][single_index]}"
        )
        print(
            f"FEM Mean Divergence of index {single_index} = {dataset_results['FEM']['divergences_scalars_standard'][single_index]}"
        )

        return
    if "Div" in proposed_model_name:
        best_divergence_indices = np.where(
            dataset_results[proposed_model_name]["divergences_scalars_standard"]
            < dataset_results["FEM"]["divergences_scalars_standard"]
        )

        best_topk_divergence_indices = topk_indices(
            dataset_results[proposed_model_name][
                "divergences_scalars_standard"
            ][best_divergence_indices],
            topk,
            order="low",
        )
        print(best_topk_divergence_indices)
        best_divergence_samples_folder = figures_folder_path / "best_divergence"
        best_divergence_samples_folder.mkdir(parents=True, exist_ok=True)
        _plot_sample_comparisons(
            root_folder_path=best_divergence_samples_folder,
            meshes=meshes,
            dataset_results=dataset_results,
            sample_indices=best_topk_divergence_indices,
            baseline_model_name=baseline_model_name,
            proposed_model_name=proposed_model_name,
        )

    abs_losses = np.abs(
        dataset_results[baseline_model_name]["losses_standard"]
        - dataset_results[proposed_model_name]["losses_standard"]
    )

    highest_k_differences_baseline_proposed = topk_indices(
        abs_losses, topk, "high"  # MODIFIED
    )
    highest_k_differences_baseline_proposed_folder = (
        figures_folder_path / "highest_k_differences"
    )
    highest_k_differences_baseline_proposed_folder.mkdir(
        parents=True, exist_ok=True
    )
    for kind_k in ("worst", "best"):
        k_predictions_folder = figures_folder_path / f"{kind_k}_k_predictions"
        model_name = proposed_model_name
        # model_name = baseline_model_name
        k_predictions_folder.mkdir(parents=True, exist_ok=True)
        k_predictions_indices = topk_indices(
            dataset_results[model_name]["losses_standard"],
            topk,
            "high" if kind_k == "worst" else "low",  # MODIFIED
        )
        _plot_sample_comparisons(
            root_folder_path=k_predictions_folder,
            meshes=meshes,
            dataset_results=dataset_results,
            sample_indices=k_predictions_indices,
            baseline_model_name=baseline_model_name,
            proposed_model_name=proposed_model_name,
        )
    _plot_sample_comparisons(
        root_folder_path=highest_k_differences_baseline_proposed_folder,
        meshes=meshes,
        dataset_results=dataset_results,
        sample_indices=highest_k_differences_baseline_proposed,
        baseline_model_name=baseline_model_name,
        proposed_model_name=proposed_model_name,
    )


if __name__ == "__main__":
    Fire(main)
