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

import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import torch_geometric as PyG
import yaml
from fire import Fire
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gnn_local_stress import data_utils, datasets, models
from gnn_local_stress.datasets import NodeType

SEED = 69


def normalized_mse_loss_single(
    ground_truth_local_stress: torch.Tensor,
    predicted_local_stress: torch.Tensor,
) -> float | torch.Tensor:
    mean_gt = ground_truth_local_stress.mean(axis=0)  # Shape == (1,3)
    mse = (
        (ground_truth_local_stress - predicted_local_stress)
        .square()
        .sum(axis=0)  # .sum(axis=0)
    )  # Shape == (N, 3)
    normalization_term = (
        (ground_truth_local_stress - mean_gt)
        .square()
        .sum(axis=0)  # .sum(axis=0)
    )  # Shape == (N, 3)
    loss = (mse / normalization_term).mean()  # Shape (1)
    return loss


def compute_divergence(
    local_stress_field: torch.Tensor,
    op_div_matrix: torch.Tensor,
    surface_nodes_ids: torch.Tensor,
    reduce_strategy: str = "square",
) -> torch.Tensor:
    if reduce_strategy not in ("abs", "square"):
        raise AttributeError("reduce_strategy must be 'abs' or 'square'")
    stress_x_xy = local_stress_field[:, [0, 2]].T.reshape(-1)
    stress_xy_y = local_stress_field[:, [2, 1]].T.reshape(-1)
    stress_x_xy_xy_y = torch.stack([stress_x_xy, stress_xy_y], axis=1)  # 2Nx2
    # Weird indexing for op_div_matrix is due to a side effect on tensor
    # compression done by torch geometric.
    div_sigma = (
        op_div_matrix.to_dense()[:, : op_div_matrix.shape[0] * 2]
        @ stress_x_xy_xy_y
    )
    # div_sigma = op_div_matrix @ stress_x_xy_xy_y
    # div_sigma.shape == (N,2)
    external_boundary_nodes_mask = (
        surface_nodes_ids == NodeType.EXTERNAL_BOUNDARY
    ).squeeze()
    internal_boundary_nodes_mask = (
        surface_nodes_ids == NodeType.INTERNAL_BOUNDARY
    ).squeeze()
    div_sigma[external_boundary_nodes_mask] = 0
    div_sigma[internal_boundary_nodes_mask] = 0
    if reduce_strategy == "abs":
        div_sigma = torch.abs(div_sigma)
    elif reduce_strategy == "square":
        div_sigma = torch.square(div_sigma)
    div_sigma = torch.mean(div_sigma, axis=0)  # div_sigma.shape == 2
    return torch.sum(div_sigma)  # Tensor[float] Scalar


def train(
    model: models.StressFieldBaseModel,
    train_dataloader: PyG.loader.DataLoader,
    test_dataloader: PyG.loader.DataLoader,
    epochs: int,
    tensorboard_writer: SummaryWriter,
    criterion: torch.nn.Module,
    learning_rate: float = 0.001,
    weights_folder: str = "",
    device: str = "cpu",
    early_stopping_limit: int = 10,
    optimize_divergence: bool = True,
    divergence_penalty: float = 1.0,
    train_all_epochs: bool = False,
    monitor_divergence_in_test: bool = False,
) -> tuple[list[float], list[float]]:
    gradient_scaler = torch.cuda.amp.GradScaler()
    train_logs_folder = Path(weights_folder)
    train_logs_folder.mkdir(parents=True, exist_ok=False)
    model_weights_path = train_logs_folder / "model_weights.pth"
    last_epoch_model_weights_path = (
        train_logs_folder / "last_epoch_model_weights.pth"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    train_parameters = ""
    train_parameters += f"Device = {device};\n"
    train_parameters += f"Batch size = {train_dataloader.batch_size};\n"
    train_parameters += f"Learning rate = {learning_rate};\n"
    train_parameters += f"Epochs = {epochs};\n"
    train_parameters += (
        f"Weights path = {str(model_weights_path.absolute())};\n"
    )
    train_parameters += f"Loss function = {str(criterion)};\n"
    train_parameters += f"Optimize divergence = {optimize_divergence};\n"
    train_parameters += f"Divergence lamba = {divergence_penalty};\n"
    train_parameters += f"Early stopping limit = {early_stopping_limit};\n"

    tensorboard_writer.add_text("Train parameters", train_parameters)
    best_loss = sys.float_info.max  # Best loss init for save best model
    train_losses: list[float] = []
    test_losses: list[float] = []
    early_stopping_counter = 0
    for epoch in range(epochs):
        # Use early stopping only when train_all_epochs is set to false
        if (
            not train_all_epochs
            and early_stopping_counter >= early_stopping_limit
        ):
            print("Training early stopped")
            break
        train_epoch_nmse_loss = 0
        train_epoch_divergence_loss = 0
        train_epoch_total_loss = 0  # NMSE + lambda*div(pred_sigma)
        test_epoch_nmse_loss = 0
        test_epoch_divergence_loss = 0
        test_epoch_total_loss = 0  # NMSE + lambda*div(pred_sigma)
        model.train()
        train_loop = tqdm(train_dataloader, leave=True)
        for mesh_graph_batch in train_loop:
            batch_loss = 0
            batch_divergence_loss = 0
            mesh_graph_batch = mesh_graph_batch.to(device)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                predicted_local_stress = model.forward(
                    mesh_graph_batch, scale_output=False, scale_input=True
                ).local_stress
                ground_truth_local_stress = data_utils.standardize(
                    mesh_graph_batch.local_stress,
                    model.mean_local_stress,
                    model.std_local_stress,
                )
                mesh_graph_batch.local_stress = ground_truth_local_stress  # Set normalized stress to GT graph
                for (
                    mesh_graph_sample,
                    predicted_local_stress_sample,
                ) in data_utils.slice_batch_gt_and_predictions(
                    mesh_graph_batch, predicted_local_stress
                ):
                    graph_loss = criterion(
                        ground_truth_local_stress=mesh_graph_sample.local_stress,
                        predicted_local_stress=predicted_local_stress_sample,
                    )

                    batch_loss += graph_loss
                    if optimize_divergence:
                        divergence_predicted_stress = (
                            compute_divergence(
                                predicted_local_stress_sample,
                                mesh_graph_sample.op_div_matrix,
                                mesh_graph_sample.surfaces_nodes_for_div.to(
                                    device
                                ),
                                reduce_strategy="square",
                            )
                            * divergence_penalty
                        )
                        batch_divergence_loss += divergence_predicted_stress
                batch_loss /= mesh_graph_batch.batch_size
                train_epoch_nmse_loss += batch_loss.detach()
                if optimize_divergence:
                    batch_divergence_loss /= mesh_graph_batch.batch_size
                    batch_loss += batch_divergence_loss
                    train_epoch_divergence_loss += (
                        batch_divergence_loss.detach()
                    )
                # Normalize loss by batch size
                train_epoch_total_loss += batch_loss.detach()

            optimizer.zero_grad()
            gradient_scaler.scale(batch_loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
        model.eval()
        with torch.no_grad():
            test_loop = tqdm(test_dataloader, leave=True)
            for mesh_graph_batch in test_loop:
                batch_loss = 0
                batch_divergence_loss = 0
                mesh_graph_batch = mesh_graph_batch.to(device)
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    predicted_graph_batch = model.forward(
                        mesh_graph_batch, scale_output=False, scale_input=True
                    )
                predicted_local_stress = predicted_graph_batch.local_stress
                ground_truth_local_stress = data_utils.standardize(
                    mesh_graph_batch.local_stress,
                    model.mean_local_stress,
                    model.std_local_stress,
                )
                mesh_graph_batch.local_stress = ground_truth_local_stress  # Set normalized stress to GT graph
                # Monitor divergence evolution during training
                for (
                    mesh_graph_sample,
                    predicted_local_stress_sample,
                ) in data_utils.slice_batch_gt_and_predictions(
                    mesh_graph_batch, predicted_local_stress
                ):
                    graph_loss = criterion(
                        ground_truth_local_stress=mesh_graph_sample.local_stress,
                        predicted_local_stress=predicted_local_stress_sample,
                    )
                    batch_loss += graph_loss
                    if monitor_divergence_in_test:
                        divergence_predicted_stress = compute_divergence(
                            predicted_local_stress_sample,
                            mesh_graph_sample.op_div_matrix.to(device),
                            mesh_graph_sample.surfaces_nodes_for_div.to(device),
                            reduce_strategy="square",
                        )
                        batch_divergence_loss += divergence_predicted_stress
                batch_loss /= mesh_graph_batch.batch_size
                test_epoch_nmse_loss += batch_loss.detach()
                if monitor_divergence_in_test:
                    batch_divergence_loss /= mesh_graph_batch.batch_size
                    batch_loss += batch_divergence_loss
                    test_epoch_divergence_loss += batch_divergence_loss.detach()
                # Normalize loss by batch size
                test_epoch_total_loss += batch_loss.detach()

            # LOGGING STUFF
            # len(dataloader) == number of batches in dataloader
            if monitor_divergence_in_test:
                test_div_value = test_epoch_divergence_loss.item() / len(
                    test_dataloader
                )
                tensorboard_writer.add_scalar(
                    "Loss/Divergence test value", test_div_value, epoch + 1
                )
            train_mse_loss = train_epoch_nmse_loss.item() / len(
                train_dataloader
            )
            total_loss = train_epoch_total_loss.item() / len(train_dataloader)
            test_loss = test_epoch_total_loss.item() / len(test_dataloader)
            tensorboard_writer.add_scalar(
                "Loss/MSE Train", train_mse_loss, epoch + 1
            )
            tensorboard_writer.add_scalar(
                "Loss/Loss Train", total_loss, epoch + 1
            )
            tensorboard_writer.add_scalar("Loss/MSE Test", test_loss, epoch + 1)
            if optimize_divergence:
                div_term = train_epoch_divergence_loss.item() / len(
                    train_dataloader
                )
                tensorboard_writer.add_scalar(
                    "Loss/Divergence Train", div_term, epoch + 1
                )
            tensorboard_writer.flush()  # To prevent memory leaks
            if test_loss < best_loss:
                models.save_model_checkpoint(
                    model, optimizer, epoch + 1, model_weights_path.as_posix()
                )
                print(f"Checkpoint saved at {model_weights_path}")
                best_loss = test_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            loss_message_to_display = f"Epoch: {epoch+1} / {epochs}, \nTotal train Loss : {total_loss}\nMSE train Loss : {train_mse_loss} \nTest Loss : {test_loss}"
            if optimize_divergence:
                loss_message_to_display += f"\nDivergence term train {div_term}"
            print(loss_message_to_display)
            train_losses.append(total_loss)
            test_losses.append(test_loss)
    models.save_model_checkpoint(
        model, optimizer, epoch + 1, last_epoch_model_weights_path.as_posix()
    )
    print(
        f"Last checkpoint at epoch {epoch+1} saved at {last_epoch_model_weights_path}"
    )
    return train_losses, test_losses


def _write_dataset_histograms(
    dataset_dataframe: pd.DataFrame,
    tensorboard_writer: SummaryWriter,
    dataset_tag: str,
) -> None:
    tensorboard_writer.add_histogram(
        dataset_tag + "/Hole plate radius",
        dataset_dataframe["hole_plate_radius"].to_numpy(),
    )
    tensorboard_writer.add_histogram(
        dataset_tag + "/Mean stress X",
        dataset_dataframe["mean_stress_x"].to_numpy(),
    )
    tensorboard_writer.add_histogram(
        dataset_tag + "/Mean stress Y",
        dataset_dataframe["mean_stress_y"].to_numpy(),
    )
    tensorboard_writer.add_histogram(
        dataset_tag + "/Mean stress XY",
        dataset_dataframe["mean_stress_xy"].to_numpy(),
    )


def run_experience(
    dataset_train_csv: str,
    dataset_test_csv: str,
    results_folder: str,
    epochs: int,
    batch_size: int,
    divergence: bool,
    latent_size: int,
    divergence_penalty: float,
    early_stopping_limit: int,
    learning_rate: float,
    message_passing_steps: int,
    train_all_epochs: bool = False,
    device: str = "cuda",
    periodic_graph: bool = True,
    monitor_divergence_in_test: bool = False,
    config_path: Path = Path(""),
    *args: Any,
    **kwargs: Any,
) -> None:
    criterion = normalized_mse_loss_single
    print(f"DATASET TRAIN CSV {dataset_train_csv}")
    print(f"DATASET TEST CSV {dataset_test_csv}")

    print(f"EPOCHS {epochs}")
    print(f"BATCH SIZE {batch_size}")
    print(f"LEARNING RATE {learning_rate}")
    print(f"LOSS function {criterion}")
    print(f"Periodic graph {periodic_graph}")
    torch.manual_seed(SEED)
    random.seed(SEED)  # Python
    np.random.seed(SEED)
    train_df = pd.read_csv(dataset_train_csv)
    test_df = pd.read_csv(dataset_test_csv)
    with SummaryWriter(log_dir=f"{results_folder}/Dataset stats") as writer:
        _write_dataset_histograms(
            tensorboard_writer=writer,
            dataset_dataframe=train_df,
            dataset_tag="Train Dataset",
        )
        _write_dataset_histograms(
            tensorboard_writer=writer,
            dataset_dataframe=test_df,
            dataset_tag="Test Dataset",
        )
        writer.add_text("Dataset train csv", dataset_train_csv)
        writer.add_text("Dataset test csv", dataset_test_csv)

    print(f"Size train dataset {len(train_df)}")
    print(f"Size test dataset {len(test_df)}")
    print("Loading datasets...")
    train_dataset = datasets.MeshStressFieldDatasetInMemory(
        train_df,
        periodic_graph=periodic_graph,
    )
    test_dataset = datasets.MeshStressFieldDatasetInMemory(test_df)
    train_dataloader = PyG.loader.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = PyG.loader.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    input_nodes_features_size = 6
    output_nodes_features_size = 3
    model = models.EncodeProcessDecode(
        input_edges_features_size=1,
        input_nodes_features_size=input_nodes_features_size,
        message_passing_steps=message_passing_steps,
        latent_size=latent_size,
        output_nodes_features_size=output_nodes_features_size,
        mean_mean_stress=train_dataset.mean_mean_stress.to(device),
        std_mean_stress=train_dataset.std_mean_stress.to(device),
        mean_local_stress=train_dataset.mean_local_stress.to(device),
        std_local_stress=train_dataset.std_local_stress.to(device),
        mean_pos=train_dataset.mean_pos.to(device),
        std_pos=train_dataset.std_pos.to(device),
        mean_edge_weight=train_dataset.mean_edge_weight.to(device),
        std_edge_weight=train_dataset.std_edge_weight.to(device),
    )

    model_summary = models.print_model(model, train_dataloader, device)
    model.to(device)
    train_logdir = results_folder + "/train_logs/"
    shutil.copyfile(config_path, Path(results_folder) / config_path.name)
    model_weights_folder = results_folder + "/weights/"
    with SummaryWriter(log_dir=train_logdir) as writer:
        writer.add_text("Model summary", model_summary)
        train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            learning_rate=learning_rate,
            epochs=epochs,
            tensorboard_writer=writer,
            criterion=criterion,
            weights_folder=model_weights_folder,
            early_stopping_limit=early_stopping_limit,
            divergence_penalty=divergence_penalty,
            optimize_divergence=divergence,
            device=device,
            train_all_epochs=train_all_epochs,
            monitor_divergence_in_test=monitor_divergence_in_test,
        )


def main(config_path: str) -> None:
    pv.start_xvfb()
    with open(config_path, "r") as file:
        experience_params = yaml.safe_load(file)
    experience_params["config_path"] = Path(config_path)
    run_experience(**experience_params)


if __name__ == "__main__":
    Fire(main)
