"""Periodicity check for 2D meshes.

Drop-in replacement for `microgen.mesh.is_periodic` so we can drop the
microgen dependency (which pulls in cadquery -> cadquery-ocp -> vtk==9.3.1,
blocking newer Python versions).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def is_periodic(
    nodes_coords: npt.NDArray[np.float64],
    tol: float = 1e-8,
) -> bool:
    """Check whether a 2D mesh is periodic in both X and Y.

    A mesh is periodic if nodes on the minimum boundary (x- / y-) have
    matching counterparts on the maximum boundary (x+ / y+) at the same
    transverse position.

    Parameters
    ----------
    nodes_coords : (N, 2) array
        Node coordinates (2D only).
    tol : float
        Position matching tolerance.

    Returns
    -------
    bool
        True if periodic, False otherwise.
    """
    if nodes_coords.shape[1] != 2:
        raise ValueError("Only 2D meshes are supported")

    min_pt = nodes_coords.min(axis=0)
    max_pt = nodes_coords.max(axis=0)

    for axis in (0, 1):
        transverse = 1 - axis
        min_mask = np.abs(nodes_coords[:, axis] - min_pt[axis]) < tol
        max_mask = np.abs(nodes_coords[:, axis] - max_pt[axis]) < tol

        if min_mask.sum() != max_mask.sum():
            return False

        min_sorted = np.sort(nodes_coords[min_mask, transverse])
        max_sorted = np.sort(nodes_coords[max_mask, transverse])

        if np.abs(max_sorted - min_sorted).max() > tol:
            return False

    return True
