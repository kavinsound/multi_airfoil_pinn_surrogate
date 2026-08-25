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
        if key in ["gammaInt", "k", "nut", "omega", "p", "phi", "ReThetat", "U"]:
            internal_data[key] = internalMesh.cell_data[key]

    for key in boundary[0][2].point_data.keys():
        if (key in ["wallShearStress", "Cp"]):
            boundary_data[key] = boundary[0][2].point_data[key]

    #converting to 2d

    internal_mask = np.isclose(internal_coords[:, 2], 0)
    internal_coords = internal_coords[internal_mask, :2]

    for name, array in internal_data:
        internal_data[name] = array[internal_mask]

    internal_data["U"] = internal_data["U"][:, :2]

    boundary_mask = np.isclose(boundary_coords[:, 2], 0)
    boundary_coords = boundary_coords[boundary_mask, :2]

    for name, array in boundary_data:
        boundary_data[name] = array[boundary_mask]

    boundary_data["wallShearStress"] = boundary_data["wallShearStress"][:, :2]

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