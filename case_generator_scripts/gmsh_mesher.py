import glob
import shutil
import os
import sys
from pathlib import Path

import gmsh
import meshio
import numpy as np


def createMesh(airfoil_list, y_plus_list, target_case_dir):
    if not gmsh.is_initialized():
        gmsh.initialize(sys.argv)
    else:
        gmsh.clear()
    gmsh.model.add("airfoil mesh") #initialize

    #Mesh resolution
    #lc_airfoil = 1e-4 #meters (around 100x smallest layer)
    lc_farfield = 0.05 #10 cm farfield
    lc_airfoil = 0.0
    chord = 1.0

    # box bounding 
    minX = -6
    maxX = 10
    minY = -6
    maxY = 6

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

    gmsh.model.geo.synchronize()

    #adding distance based levels instead of pure refinement box
    gmsh.model.mesh.field.add("Distance", 1) #field 1 calculates distance
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", curve_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)


    gmsh.model.mesh.field.add("Threshold", 2) #sets thresholds
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 2.5e-3)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.1)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.02)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 5)

    # --- 3. Individual Boundary Layer Fields ---
    bl_field_tags = []
    for i, (foil_tag, first_layer_size) in enumerate(zip(curve_tags, y_plus_list)):
        bl_tag = 10 + i  # Assign unique field IDs starting from 10
        gmsh.model.mesh.field.add("BoundaryLayer", bl_tag)
        
        # Apply specific configuration to this exact curve
        gmsh.model.mesh.field.setNumbers(bl_tag, "CurvesList", [foil_tag])
        gmsh.model.mesh.field.setNumber(bl_tag, "Size", first_layer_size)  # Individual first layer height
        gmsh.model.mesh.field.setNumber(bl_tag, "Ratio", 1.15)             # Growth rate
        gmsh.model.mesh.field.setNumber(bl_tag, "Thickness", 0.005)         # Absolute max thickness
        gmsh.model.mesh.field.setNumber(bl_tag, "Quads", 1)                # Structured quads
        gmsh.model.mesh.field.setAsBoundaryLayer(bl_tag) 
        bl_field_tags.append(bl_tag)

    # --- 4. Combine All Fields using a Min Operator ---
    # Field 2 (background threshold) + all individual boundary layer fields
    min_field_tag = 100
    gmsh.model.mesh.field.add("Min", min_field_tag)
    gmsh.model.mesh.field.setNumbers(min_field_tag, "FieldsList", [2] + bl_field_tags)

    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_tag)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    #set domain names
    gmsh.model.addPhysicalGroup(1, curve_tags, name="airfoils")
    gmsh.model.addPhysicalGroup(1, [c_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(1, [c_bottom, c_outlet, c_top], name="outlet")
    gmsh.model.addPhysicalGroup(2, [fluid_surface], name="FluidDomain")

    # gmsh.fltk.initialize()
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2) #generate actual mesh

    # gmsh.fltk.run()

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.write("temp.msh")

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run() 


    gmsh.finalize()
    #read with meshio and write to case
#     mesh = meshio.read("temp.msh")


#     target = os.path.join(target_case_dir, "constant", "polyMesh")

#     if target.exists():
#         shutil.rmtree(target)

# # Recreate the empty directory
#     target.mkdir(parents=True, exist_ok=True)
#     if not os.path.exists(target):
#         os.makedirs(target)

#     mesh.write(target, file_format = "openfoam")

#     if os.path.exists("temp.msh"):
#         os.remove("temp.msh")





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
    config_path = Path("../test_config").resolve()

    dat_files = config_path.glob("*.dat")
    airfoil_list = []
    for file in dat_files:
        coords = np.loadtxt(file)
        airfoil_list.append(coords)

    y_plus_path = os.path.join(config_path, "y_plus")

    y_p = np.loadtxt(y_plus_path, dtype=np.float64)
    # print(y_p)

    target_case = Path("../sample_case").resolve()
    createMesh(airfoil_list, y_p, target_case)