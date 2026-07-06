from pathlib import Path
import glob
import numpy as np
import os
from shapely.geometry import Polygon
from shapely.affinity import rotate, scale
import trimesh
import sys
import math
import matplotlib.pyplot as plt
from shapely import make_valid


#                length of id array is number of airfoils (1-3)
def mesh_polygon(ids, lengths, deflection_angles, gap_vectors, angle_of_attack, reflection):
    n = len(ids)

    file_list = dat_list()

    if (n == 1):
        foil_coords = np.loadtxt(file_list[ids[0]])
        shape = Polygon(foil_coords)
        
        shape = rotate(shape, angle=-1*np.rad2deg(angle_of_attack), origin=(0.25, 0))

        if reflection:
            shape = scale(shape, xfact=1, yfact=-1, origin=(0, 0))
        
        return shape
    
    else:
        shapes = []
        end_point = [1, 0]
        for i in range(n):
            foil_coords = np.loadtxt(file_list[ids[i]])

            if i != 0:
                foil_coords += end_point + gap_vectors[i-1]
                end_point += gap_vectors[i-1] + [lengths[i], 0]
            

            shape = Polygon(foil_coords)
            shapes.append(shape)









def dat_list():
    folder_path = Path("../cleaned_foils").resolve()

    unsorted = folder_path.glob("*.dat")
    return sorted(unsorted)
