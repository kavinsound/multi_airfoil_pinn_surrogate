import os
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import gmsh
import numpy as np


def createMesh(airfoil_list, y_plus_list, target_case_dir):
    if not gmsh.is_initialized():
        gmsh.initialize(sys.argv)
    else:
        gmsh.clear()
    gmsh.model.add("airfoil mesh") #initialize

    y_plus_list = np.atleast_1d(y_plus_list)

    #Mesh resolution
    #lc_airfoil = 1e-4 #meters (around 100x smallest layer)
    lc_farfield = 0.05 #10 cm farfield
    lc_airfoil = 0.0

    # box bounding 
    minX = -6
    maxX = 10
    minY = -6
    maxY = 6

    # num_threads = 4  # Set to your desired core count
    # gmsh.option.setNumber("Mesh.MaxNumThreads1D", num_threads) parallel meshing breaks
    # gmsh.option.setNumber("Mesh.MaxNumThreads2D", num_threads)
    # gmsh.option.setNumber("Mesh.MaxNumThreads3D", num_threads)

    curve_tags = []
    curve_loops = []
    for i, foil in enumerate(airfoil_list):
        tag, loop = createFoil(foil, lc_airfoil, f"airfoil{i+1}") 
        curve_tags.append(tag)
        curve_loops.append(loop)

   # defining box 
    p1 = gmsh.model.geo.addPoint(minX, minY, 0, lc_farfield)
    p2 = gmsh.model.geo.addPoint(maxX, minY, 0, lc_farfield)
    p3 = gmsh.model.geo.addPoint(maxX, maxY, 0, lc_farfield)
    p4 = gmsh.model.geo.addPoint(minX, maxY, 0, lc_farfield)

    c_bottom = gmsh.model.geo.addLine(p1, p2)
    c_outlet = gmsh.model.geo.addLine(p2, p3)
    c_top = gmsh.model.geo.addLine(p3, p4)
    c_inlet = gmsh.model.geo.addLine(p4, p1)

    farfield_loop = gmsh.model.geo.addCurveLoop([c_bottom, c_outlet, c_top, c_inlet])

    fluid_surface = gmsh.model.geo.addPlaneSurface([farfield_loop] + curve_loops)


    # --- EXTRUDE 2D SURFACE TO 3D ---
    # Extrude the fluid surface along the Z-axis by 0.01 units, with 1 layer, and recombine quads/hexes
    extrusion_tags = gmsh.model.geo.extrude(
        [(2, fluid_surface)], 
        dx=0.0, dy=0.0, dz=1, 
        numElements=[1], 
        heights=[],
        recombine=True
    )
    
    gmsh.model.geo.synchronize()


    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Extract the resulting 3D volume tag(s)
    volumes = [tag for dim, tag in extrusion_tags if dim == 3]         # [1]
    lateral_surfaces = [tag for dim, tag in extrusion_tags if dim == 2] # boundaries/walls
    # --- 2. Distance and Threshold Fields ---
    gmsh.model.mesh.field.add("Distance", 1)  # Field 1 calculates distance
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", curve_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 300)

    gmsh.model.mesh.field.add("Threshold", 2)  # Sets thresholds
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 5e-4)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 5e-3)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.005)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.1)
    gmsh.model.mesh.field.setNumber(2, "StopAtDistMax", 1)

    gmsh.model.mesh.field.add("Threshold", 3)  # Sets thresholds
    gmsh.model.mesh.field.setNumber(3, "InField", 1)
    gmsh.model.mesh.field.setNumber(3, "SizeMin", 5e-3)
    gmsh.model.mesh.field.setNumber(3, "SizeMax", 0.05)
    gmsh.model.mesh.field.setNumber(3, "DistMin", 1)
    gmsh.model.mesh.field.setNumber(3, "DistMax", 6) 

    # --- 3. Individual Boundary Layer Fields ---
    bl_field_tags = []

    print(curve_tags)
    print(y_plus_list)

    for i, (foil_tag, first_layer_size) in enumerate(zip(curve_tags, y_plus_list)):
        bl_tag = 10 + i  # Assign unique field IDs starting from 10
        gmsh.model.mesh.field.add("BoundaryLayer", bl_tag)
        
        # Apply specific configuration to this exact curve
        gmsh.model.mesh.field.setNumbers(bl_tag, "CurvesList", [foil_tag])
        gmsh.model.mesh.field.setNumber(bl_tag, "Size", first_layer_size)  # First layer height
        gmsh.model.mesh.field.setNumber(bl_tag, "Ratio", 1.2)             # Growth rate
        gmsh.model.mesh.field.setNumber(bl_tag, "Thickness", 0.001)        # Absolute max thickness
        gmsh.model.mesh.field.setNumber(bl_tag, "Quads", 1)                # Structured quads
        # gmsh.model.mesh.field.setNumber(bl_tag, "IntersectAxis", 1)
        gmsh.model.mesh.field.setAsBoundaryLayer(bl_tag) 
        bl_field_tags.append(bl_tag)

    # --- 4. Combine All Fields using a Min Operator ---
    min_field_tag = 100
    gmsh.model.mesh.field.add("Min", min_field_tag)
    gmsh.model.mesh.field.setNumbers(min_field_tag, "FieldsList", [2, 3] + bl_field_tags)

    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_tag)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- 5. Set Domain Names / Physical Groups ---
    # gmsh.model.addPhysicalGroup(1, curve_tags, name="airfoils")
    # gmsh.model.addPhysicalGroup(1, [c_inlet], name="inlet")
    # gmsh.model.addPhysicalGroup(1, [c_bottom, c_outlet, c_top], name="outlet")
    gmsh.model.addPhysicalGroup(3, volumes, name="FluidDomain")

    # for i, tag in enumerate(lateral_surfaces):
    #     gmsh.model.addPhysicalGroup(2, [tag], name=f"surface{i+1}")

    gmsh.model.addPhysicalGroup(2, [lateral_surfaces[0], fluid_surface], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, lateral_surfaces[1:4], name="outlets")
    gmsh.model.addPhysicalGroup(2, [lateral_surfaces[4]], name="inlet")
    gmsh.model.addPhysicalGroup(2, lateral_surfaces[5:], name="airfoils")

    gmsh.option.setNumber('Mesh.RecombineAll', 1)
    gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)

    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)


    gmsh.option.setNumber("Mesh.Smoothing", 5)

    # --- 6. Generate Mesh & Export ---
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(3)  # Generate actual 3D mesh


    # gmsh.model.mesh.optimize("Netgen")

    # gmsh.model.mesh.setOrder(2)

    # gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    # gmsh.model.mesh.optimize("HighOrder")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.write("airfoil_mesh.msh")
    mesh_path = Path("airfoil_mesh.msh")

    # if '-nopopup' not in sys.argv:
    # # Optional: explicitly initialize the GUI toolkit if it hasn't been started
    #     gmsh.fltk.initialize()
    #     print("Launching Gmsh GUI. Close the window to continue execution...")
    #     gmsh.fltk.run()

    gmsh.finalize()

    cmd = ["gmshToFoam", str(mesh_path), "-case", str(target_case_dir)]

    try:
        # Run the command and wait for it to complete
        result = subprocess.run(
            cmd,
            check=True,          # Raises CalledProcessError if the command fails
            text=True,           # Decodes stdout/stderr as strings
            capture_output=True  # Captures terminal output so we can inspect it if needed
        )
        print(f"Successfully converted mesh for case: {target_case_dir.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running gmshToFoam for {target_case_dir.name}:")
        print(e.stderr)
        raise


    log_path = Path(target_case_dir) / "meshCheck.log"

    changeDictCmd = ["changeDictionary", "-case", str(target_case_dir)]
    result = subprocess.run(
        changeDictCmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# Build the command as a list (without the '>' redirection)
    meshCheckcmd = ["checkMesh", "-allGeometry", "-writeSets", "vtk", "-allTopology", "-case", str(target_case_dir)]

    # Run the process and write stdout/stderr directly to the log file
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            meshCheckcmd, 
            check=True,
            stdout=log_file, 
            stderr=subprocess.STDOUT, # Combines error messages into the same log
            text=True
        )

    



def createFoil(coords, lc, prefix):
    # 1. If the first and last points are identical, remove the last one 
    # so the array doesn't double-tap the trailing edge node.
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    node_tags = []
    for x, y in coords:
        node_tags.append(gmsh.model.geo.addPoint(x, y, 0, lc))
    
    # To make a single spline form a closed loop, append the FIRST node tag 
    # to the end of the node_tags list so it loops back cleanly.
    node_tags.append(node_tags[0])

    curve_tag = gmsh.model.geo.addSpline(node_tags)
    curve_loop = gmsh.model.geo.addCurveLoop([curve_tag])

    gmsh.model.addPhysicalGroup(1, [curve_tag], name=f"{prefix}")
    
    # Returns a single tag and a single loop
    return curve_tag, curve_loop


if __name__ == "__main__":

    import time
    start = time.perf_counter()
    
    config_path = Path("../test_config").resolve()

    dat_files = config_path.glob("*.dat")
    airfoil_list = []
    for file in dat_files:
        coords = np.loadtxt(file)
        airfoil_list.append(coords)

    y_plus_path = os.path.join(config_path, "y_plus")

    y_p = np.loadtxt(y_plus_path, dtype=np.float64)
    # print(type(y_p))
    
    target_case = Path("../sample_case").resolve()
    createMesh(airfoil_list, y_p, target_case)

    end = time.perf_counter()

    elapsed = end - start

    print(f"Completed in {elapsed:2f} sec.")
