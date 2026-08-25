import pyvista as pv
from pathlib import Path
import numpy as np
import h5py


def readCase(path, h5_file_path):
    case_path = Path(path)
    case_name = case_path.stem
    folders = [p for p in (case_path / "VTK").iterdir() if p.is_dir()]

    folder = folders[0]

    internalMesh = pv.read(folder / "internal.vtu")
    boundary = pv.read(folder / "boundary.vtm")

    internal_coords = internalMesh.cell_centers().points
    internal_data = {}

    boundary_coords = boundary[0][2].points
    boundary_data = {}

    internalMesh = internalMesh.compute_cell_sizes()

    for key in internalMesh.cell_data.keys():
        if key in ["gammaInt", "k", "nut", "omega", "p", "phi", "ReThetat", "U", "Volume"]:
            internal_data[key] = internalMesh.cell_data[key]

    internal_data["Area"] = internal_data.pop("Volume")

    for key in boundary[0][2].point_data.keys():
        if (key in ["wallShearStress", "Cp"]):
            boundary_data[key] = boundary[0][2].point_data[key]

    #converting to 2d

    # internal_mask = np.isclose(internal_coords[:, 2], 0)
    # internal_coords = internal_coords[internal_mask, :2]

    _, internal_mask = np.unique(internal_coords, axis=0, return_index=True)

    internal_coords = internal_coords[internal_mask, :2]

    for name, array in internal_data.items():
        internal_data[name] = array[internal_mask]

    internal_data["U"] = internal_data["U"][:, :2]

    boundary_mask = np.isclose(boundary_coords[:, 2], 0)
    boundary_coords = boundary_coords[boundary_mask, :2]

    for name, array in boundary_data.items():
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

    with h5py.File(h5_file_path, "a") as f:
        # Create or clear the group for this specific case
        if case_name in f:
            del f[case_name]
        
        case_grp = f.create_group(case_name)
        
        # --- Internal Data Group ---
        internal_grp = case_grp.create_group("internal")
        internal_grp.create_dataset("coords", data=internal_coords, dtype=np.float32)
        internal_grp.create_dataset("sdf", data=sdf, dtype=np.float32)
        
        for name, array in internal_data.items():
            internal_grp.create_dataset(name, data=array)

        # --- Boundary Data Group ---
        boundary_grp = case_grp.create_group("boundary")
        boundary_grp.create_dataset("coords", data=boundary_coords, dtype=np.float32)
        
        for name, array in boundary_data.items():
            boundary_grp.create_dataset(name, data=array)


if __name__ == "__main__":
    for i in range(1,5):
        sample_target = f"../generated_cases/case_{i}"

        h5_file = Path("../sample_h5.h5")

        readCase(sample_target, h5_file)