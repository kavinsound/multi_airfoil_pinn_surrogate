import gmsh
import os
from pathlib import Path
import glob
import numpy as np
import sys
def createMesh(airfoil_list, y_plus_list, target_case_dir):
    if not gmsh.is_initialized():
        gmsh.initialize(sys.argv)
    else:
        gmsh.clear()
    gmsh.model.add("airfoil mesh") #initialize

    #Mesh resolution
    #lc_airfoil = 1e-4 #meters (around 100x smallest layer)
    lc_farfield = 0.1 #10 cm farfield
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
    gmsh.model.mesh.field.setNumbers(1, "Sampling", 100)


    gmsh.model.mesh.field.add("Threshold", 2) #sets thresholds
    gmsh.model.mesh.field.setNumber(2, "Infield", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 1e-4)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.1)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.02)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 5)

    gmsh.model.mesh.field.setAsBackground(2) #set as background mesh

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    #set domain names
    gmsh.model.addPhysicalGroup(1, curve_tags, name="airfoils")
    gmsh.model.addPhysicalGroup(1, [c_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(1, [c_bottom, c_outlet, c_top], name="outlet")
    gmsh.model.addPhysicalGroup(2, [fluid_surface], name="FluidDomain")

    gmsh.model.mesh.generate(2) #generate actual mesh

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  




def createFoil(coords, lc, prefix): #function to create individual foil loops
    node_tags = []
    for x, y in coords:
        node_tags.append(gmsh.model.geo.addPoint(x, y, 0, lc))
    curve_tag = gmsh.model.geo.addSpline(node_tags)
    curve_loop = gmsh.model.geo.addCurveLoop([curve_tag])

    gmsh.model.addPhysicalGroup(1, [curve_tag], name=f"{prefix}")
    return curve_tag, curve_loop