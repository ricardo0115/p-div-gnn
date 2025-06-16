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
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric as PyG
import yaml
from fire import Fire
from tqdm import tqdm

from gnn_local_stress import data_utils, datasets, models


def copy_data_file_and_replace_local_stress_field(
    original_data_path: str,
    target_data_path: str,
    local_stress_field: torch.Tensor,
) -> None:
    shutil.copyfile(original_data_path, target_data_path)
    org_data = dict(np.load(original_data_path))
    org_data["stress_field"] = local_stress_field.numpy()
    np.savez(target_data_path, **org_data)


def predict_and_save(
    model: models.EncodeProcessDecode,
    dataloader: PyG.loader.DataLoader,
    results_folder: Path,
    device: str,
) -> list[str]:
    fields_folder = results_folder / "fields"
    fields_folder.mkdir(exist_ok=True, parents=True)
    mesh_id = 0
    predicted_data_filenames: list[str] = []
    for mesh_graph_batch in tqdm(dataloader, leave=True):
        mesh_graph_batch = mesh_graph_batch.to(device)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            predicted_graph_batch = model.forward(
                mesh_graph_batch, scale_output=True, scale_input=True
            )
        for predicted_stress_field in data_utils.slice_batch_predictions(
            batch_graph_prediction=predicted_graph_batch.local_stress,
            batch_indices=mesh_graph_batch.batch,
        ):

            field_filename = f"hole_plate_mesh_{str(mesh_id)}.npz"
            predicted_data_path = (fields_folder / field_filename).as_posix()
            original_data_path = dataloader.dataset.dataframe.data_filename[
                mesh_id
            ]
            copy_data_file_and_replace_local_stress_field(
                original_data_path=original_data_path,
                target_data_path=predicted_data_path,
                local_stress_field=predicted_stress_field.cpu(),
            )

            mesh_id += 1
            predicted_data_filenames.append(predicted_data_path)
            # Store field in a .npz file (same as in data generation)
            # Store .csv file that matches .vtk file and .npz data file
    return predicted_data_filenames


@torch.no_grad()
def run_inference(
    dataset_csv: str | Path,
    results_folder: str | Path,
    model_weights_path: str | Path,
    periodic_graph: bool,
    batch_size: int,
    latent_size: int,
    message_passing_steps: int,
    device: str,
    config_path: Path,
) -> None:
    print(f"DATASET CSV PATH {dataset_csv}")
    print(f"INFERENCE FOLDER {results_folder}")
    dataframe = pd.read_csv(dataset_csv)
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    # Copy config file to results for tracking
    shutil.copyfile(config_path, Path(results_folder) / config_path.name)
    print("Loading datasets...")
    dataset = datasets.MeshStressFieldDatasetInMemory(
        dataframe,
        periodic_graph=periodic_graph,
    )
    dataloader = PyG.loader.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # THE DATASET MUST NOT BE SHUFFLED
    )
    input_nodes_features_size = 6
    output_nodes_features_size = 3
    model = models.EncodeProcessDecode(
        input_edges_features_size=1,
        input_nodes_features_size=input_nodes_features_size,
        message_passing_steps=message_passing_steps,
        latent_size=latent_size,
        output_nodes_features_size=output_nodes_features_size,
    )
    model.to(device)
    models.load_model_checkpoint(model, str(model_weights_path))
    models.print_model(model, dataloader, device)
    model.eval()
    print("Running inferences...")
    predicted_data_filenames = predict_and_save(
        model, dataloader, results_folder, device
    )
    dataframe["data_filename"] = predicted_data_filenames
    dataframe.to_csv((results_folder / "dataset.csv").as_posix(), index=False)
    standardization_parameters = {
        "mean_local_stress": float(model.mean_local_stress.cpu()),
        "std_local_stress": float(model.std_local_stress.cpu()),
    }
    standardization_path = (results_folder / "normalize_params.json").as_posix()
    with open(standardization_path, "w") as file:
        json.dump(standardization_parameters, file)


def main(config_path: str) -> None:
    with open(config_path, "r") as file:
        inference_params = yaml.safe_load(file)
    inference_params["config_path"] = Path(config_path)
    run_inference(**inference_params)


if __name__ == "__main__":
    Fire(main)
