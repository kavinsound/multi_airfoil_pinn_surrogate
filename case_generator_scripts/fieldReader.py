import pyvista as pv
from pathlib import Path
import numpy as np
import h5py


def readCase(path):
    case_path = Path(path)

    folders = [p for p in (case_path / "VTK").iterdir() if p.is_dir()]

    folder = folders[0]

    internalMesh = pv.read(folder / "internal.vtu")
    boundary = pv.read(folder / "boundary.vtm")

    internal_coords = internalMesh.cell_centers().points
    internal_data = {}

    boundary_coords = boundary[0][2].points
    boundary_data = {}

    for key in internalMesh.cell_data.keys():
        if ("Mean" in key or "Prime2Mean" in key) and "wallShearStress" not in key and "Cp" not in key:
            internal_data[key] = internalMesh.cell_data[key]

    for key in boundary[0][2].point_data.keys():
        if (key in ["wallShearStressMean", "CpMean"]):
            boundary_data[key] = boundary[0][2].point_data[key]

    #converting to 2d

    internal_mask = np.isclose(internal_coords[:, 2], 0)
    internal_coords = internal_coords[internal_mask, :2]

    for name, array in internal_data:
        internal_data[name] = array[internal_mask]

    internal_data["UMean"] = internal_data["UMean"][:, :2]
    internal_data["UPrime2Mean"] = internal_data["UPrime2Mean"][:, [0, 1, 3]] #remove z directions

    boundary_mask = np.isclose(boundary_coords[:, 2], 0)
    boundary_coords = boundary_coords[boundary_mask, :2]

    for name, array in boundary_data:
        boundary_data[name] = array[boundary_mask]

    boundary_data["wallShearStressMean"] = boundary_data["wallShearStressMean"][:, :2]

    from scipy.spatial import KDTree
    from matplotlib.path import Path as geoPath

    boundary_tree = KDTree(boundary_coords)

    distances, _ = boundary_tree.query(internal_coords)

    path = geoPath(boundary_coords)
    inside_mask = path.contains_points(internal_coords)
    distances[inside_mask] *= -1

    sdf = distances.astype(np.float32)

    #add the hd5 logic here. create organized groups


if __name__ == "__main__":
    sample_target = "../sample_case/"

    readCase(sample_target)