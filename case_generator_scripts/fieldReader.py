import pyvista as pv
from pathlib import Path
import numpy as np
import h5py


def readCase(path, h5_file_path):
    case_path = Path(path)
    case_name = case_path.stem
    print(f"reading {case_name}...")
    folders = [p for p in (case_path / "VTK").iterdir() if p.is_dir()]

    folder = folders[0]

    internalMesh = pv.read(folder / "internal.vtu")
    boundary = pv.read(folder / "boundary.vtm")

    cell_centers = internalMesh.cell_centers().points 

    # Get coordinates (2D)
    internal_coords = cell_centers[:, :2]  # Drop z coordinate

    # Get connectivity (PolyData stores faces, not cells)

    # --- Map cell data from 3D to 2D ---
    internal_data = {}

    # Find cells in the middle z-plane

    for key in internalMesh.cell_data.keys():
        if key in ["gammaInt", "k", "nut", "omega", "p", "phi", "ReThetat", "U", "Volume"]:
            # Get data for cells in the middle plane
            data = internalMesh.cell_data[key]
            
            if key == "U":
                # Keep only x,y components
                data = data[:, :2]
            
            internal_data[key] = data

    # Rename Volume to Area (since it's now 2D)
    if "Volume" in internal_data:
        internal_data["Area"] = internal_data.pop("Volume")
    
    from scipy.spatial import Delaunay
    tri = Delaunay(internal_coords)
    
    # Build edge list from triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    
    edges = np.array(list(edges), dtype=np.int32)


    boundary_coords = boundary[0][2].points
    boundary_data = {}



    for key in boundary[0][2].point_data.keys():
        if (key in ["wallShearStress", "Cp"]):
            boundary_data[key] = boundary[0][2].point_data[key]

    #converting to 2d

    # internal_mask = np.isclose(internal_coords[:, 2], 0)
    # internal_coords = internal_coords[internal_mask, :2]

    # _, internal_mask = np.unique(internal_coords, axis=0, return_index=True)

    # internal_coords = internal_coords[internal_mask, :2]

    # for name, array in internal_data.items():
    #     internal_data[name] = array[internal_mask]

    # internal_data["U"] = internal_data["U"][:, :2]

    boundary_mask = np.isclose(boundary_coords[:, 2], 0)
    boundary_coords = boundary_coords[boundary_mask, :2]

    for name, array in boundary_data.items():
        boundary_data[name] = array[boundary_mask]

    boundary_data["wallShearStress"] = boundary_data["wallShearStress"][:, :2]

    #add wallshearstress to Cf conversion here

    velocity_file = case_path / "0.orig" / "include" / "initialConditions"
    import re

    with open(velocity_file, "r") as f:

        content = f.read()
        match = re.search(r"velocity\s+([\d.]+)", content)
        if match:
            vel = float(match.group(1))

    Cf = np.linalg.norm(boundary_data.pop("wallShearStress"), axis=1) / (0.5 * vel**2)
    boundary_data["Cf"] = Cf

    #drag coefficients

    coeff_file_path = case_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    data = np.loadtxt(coeff_file_path, dtype=np.float32)
    Cd, Cl = data[:, 1], data[:, 4]

    Cd_avg, Cl_avg = np.mean(Cd[-500:]), np.mean(Cl[-500:])

    # print(Cd_avg, Cl_avg)

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
        # internal_grp.create_dataset("cells", data=cells, dtype=np.float32)
        # internal_grp.create_dataset("cell_offsets", data=cell_offsets, dtype=np.float32)

        internal_grp.create_dataset("edges", data=edges, dtype=np.int32) #connectivity data

        internal_grp.create_dataset("sdf", data=sdf, dtype=np.float32)
        
        for name, array in internal_data.items():
            internal_grp.create_dataset(name, data=array)

        # --- Boundary Data Group ---
        boundary_grp = case_grp.create_group("boundary")
        boundary_grp.create_dataset("coords", data=boundary_coords, dtype=np.float32)
        
        for name, array in boundary_data.items():
            boundary_grp.create_dataset(name, data=array)

        coeff_grp = case_grp.create_group("coeffs")
        coeff_grp.create_dataset("Cd", data=Cd_avg, dtype=np.float32)
        coeff_grp.create_dataset("Cl", data=Cl_avg, dtype=np.float32)


    # print("-" * 16)

if __name__ == "__main__":
    for i in range(1,5):
        sample_target = f"../generated_cases/case_{i}"

        h5_file = Path("../sample_h5.h5")

        readCase(sample_target, h5_file)