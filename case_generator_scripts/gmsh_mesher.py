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

    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)

    curve_tags = []
    curve_loops = []
    te_tags = []
    for i, foil in enumerate(airfoil_list):
        tag, loop, te_tag = createFoil(foil, lc_airfoil, f"airfoil{i+1}") 
        curve_tags.append(tag)
        curve_loops.append(loop)
        te_tags.append(te_tag)

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

    # gmsh.model.occ.removeAllDuplicates()
    # gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Extract the resulting 3D volume tag(s)
    volumes = [tag for dim, tag in extrusion_tags if dim == 3]         # [1]
    lateral_surfaces = [tag for dim, tag in extrusion_tags if dim == 2] # boundaries/walls
    # --- 2. Distance and Threshold Fields ---
    flat_curve_tags = [tag for pair in curve_tags for tag in pair]

    gmsh.model.mesh.field.add("Distance", 1)  # Field 1 calculates distance
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", flat_curve_tags)
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
        gmsh.model.mesh.field.setNumbers(bl_tag, "CurvesList", foil_tag)
        gmsh.model.mesh.field.setNumber(bl_tag, "Size", first_layer_size)  # First layer height
        gmsh.model.mesh.field.setNumber(bl_tag, "Ratio", 1.2)             # Growth rate
        gmsh.model.mesh.field.setNumber(bl_tag, "Thickness", 0.001)        # Absolute max thickness
        gmsh.model.mesh.field.setNumber(bl_tag, "Quads", 1)                # Structured quads

        curve_te_tags = te_tags[i]
        upper_tag = curve_te_tags[0]
        lower_tag = curve_te_tags[1]

        # if upper_tag == lower_tag:
        #     # Sharp TE: the upper/lower TE points were merged into one
        #     # point in createFoil, so there's only a single fan point.
        #     fan_points = [upper_tag]
        #     fan_sizes = [8]
        # else:
        #     # Blunt TE: two distinct fan points, bumped up from the
        #     # original [1, 1] to spread the corner transition over more
        #     # elements instead of collapsing it into a sliver.
        #     fan_points = [upper_tag, lower_tag]
        #     fan_sizes = [8, 8]
 
        # gmsh.model.mesh.field.setNumbers(bl_tag, "FanPointsList", fan_points)
        # gmsh.model.mesh.field.setNumbers(bl_tag, "FanPointsSizesList", fan_sizes)
 


        # gmsh.model.mesh.field.setNumbers(bl_tag, "FanPointsList", [upper_tag, lower_tag])
        # gmsh.model.mesh.field.setNumbers(bl_tag, "FanPointsSizesList", [3, 3])  # tune 3-6


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
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    node_tags = []
    for x, y in coords:
        node_tags.append(gmsh.model.geo.addPoint(x, y, 0, lc))

    te_upper_tag = node_tags[0]
    te_lower_tag = node_tags[-1]
    te_tags = [te_upper_tag, te_lower_tag]

    # Spline over the airfoil surface ONLY — no wraparound repeat here
    surface_spline = gmsh.model.geo.addSpline(node_tags)

    # Explicit straight line closes the loop across the blunt TE face
    te_line = gmsh.model.geo.addLine(te_lower_tag, te_upper_tag)

    # curve_tags_foil = surface_spline
    curve_tags_foil = [surface_spline, te_line]
    curve_loop = gmsh.model.geo.addCurveLoop(curve_tags_foil)
    gmsh.model.addPhysicalGroup(1, curve_tags_foil, name=f"{prefix}")

    return curve_tags_foil, curve_loop, te_tags

# def createFoil(coords, lc, prefix, first_layer_size, sharp_te_factor=3.0):
#     """
#     Build the airfoil curve loop.
 
#     If the trailing-edge gap is smaller than `sharp_te_factor` times the
#     intended first boundary-layer height, gmsh cannot physically fit a
#     real blunt-TE layer into that gap without collapsing a corner cell
#     to near-zero volume. In that case we merge the upper/lower TE points
#     into a single point and build one closed spline with no TE face —
#     this is what produced the 2e-10-volume / low-interpolation-weight
#     faces before.
 
#     Otherwise (a genuinely blunt TE), we keep the original blunt-TE
#     handling: separate upper/lower TE points joined by a straight
#     closing line.
#     """
#     if np.allclose(coords[0], coords[-1]):
#         coords = coords[:-1]
 
#     te_gap = np.linalg.norm(np.asarray(coords[0]) - np.asarray(coords[-1]))
#     sharp_te = te_gap < sharp_te_factor * first_layer_size
 
#     if sharp_te:
#         # Merge the two TE points into their midpoint and drop them from
#         # the interior point list so they aren't duplicated.
#         merged_te = (np.asarray(coords[0]) + np.asarray(coords[-1])) / 2.0
#         interior_coords = coords[1:-1]
 
#         node_tags = [gmsh.model.geo.addPoint(merged_te[0], merged_te[1], 0, lc)]
#         for x, y in interior_coords:
#             node_tags.append(gmsh.model.geo.addPoint(x, y, 0, lc))
 
#         te_tags = [node_tags[0], node_tags[0]]  # single fan point, both entries equal
 
#         # Closed spline: repeat the first point tag at the end
#         surface_spline = gmsh.model.geo.addSpline(node_tags + [node_tags[0]])
#         curve_tags_foil = [surface_spline]
 
#     else:
#         node_tags = []
#         for x, y in coords:
#             node_tags.append(gmsh.model.geo.addPoint(x, y, 0, lc))
 
#         te_upper_tag = node_tags[0]
#         te_lower_tag = node_tags[-1]
#         te_tags = [te_upper_tag, te_lower_tag]
 
#         # Spline over the airfoil surface ONLY — no wraparound repeat here
#         surface_spline = gmsh.model.geo.addSpline(node_tags)
 
#         # Explicit straight line closes the loop across the blunt TE face
#         te_line = gmsh.model.geo.addLine(te_lower_tag, te_upper_tag)
 
#         curve_tags_foil = [surface_spline, te_line]
 
#     curve_loop = gmsh.model.geo.addCurveLoop(curve_tags_foil)
#     gmsh.model.addPhysicalGroup(1, curve_tags_foil, name=f"{prefix}")
 
#     return curve_tags_foil, curve_loop, te_tags



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
