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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv("benchmark_data_elastic.csv")

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
    data["n_nodes"], data["gnn_gpu"], label="GNN (GPU)", color="C0", marker="o"
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
ax.set_title("Computation Time vs. Number of Nodes (Linear elasticity)")

# Grid and legend
ax.grid(True, which="both", ls="--", linewidth=0.5)
ax.legend()

# Tight layout for better spacing
plt.tight_layout()

# Save to file
plt.savefig("computation_time_vs_nodes_elastic.pdf", format="pdf", dpi=300)

# Show the plot
plt.show()
