from pathlib import Path
import glob
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, scale
import trimesh
import sys
import math
import matplotlib.pyplot as plt
from shapely import make_valid


#                length of id array is number of airfoils (1-3)
def mesh_polygon(ids, lengths, deflection_angles, gap_vectors, angle_of_attack, reflection):
    n = len(ids)

    file_list = dat_list() #list of foils

    if (n == 1): #only one airfoil is simple
        foil_coords = np.loadtxt(file_list[ids[0]])
        shape = Polygon(foil_coords)
        
        shape = rotate(shape, angle=-1*np.rad2deg(angle_of_attack), origin=(0.25, 0)) #rotate by attack angle

        if reflection:
            shape = scale(shape, xfact=1, yfact=-1, origin=(0, 0)) #reflect for downward
        
        return shape
    
    else:
        shapes = [] #list to hold each polygon to combine after
        end_point = [1, 0] #end of each polygon as we add to list
        for i in range(n):
            foil_coords = np.loadtxt(file_list[ids[i]])

            if i != 0:
                
                foil_coords += end_point + gap_vectors[i-1]   #after first airfoil, calculate starting point by adding the vectors to origin
                end_point += gap_vectors[i-1] + [lengths[i], 0]  #recalculate new endpoint of new part
            
            
            shape = Polygon(foil_coords)
            
            
            centroid = end_point - [3 * lengths[i]/4, 0] #calculate absolute rotation center with new endpoint
            shape = rotate(shape, angle = -1*np.rad2deg(deflection_angles[i]), origin=centroid)  #apply individual deflection angle

            start = end_point - [lengths[i], 0]   #find tip of airfoil to apply scale without moving it around
            shape = scale(shape, xfact=lengths[i], yfact=lengths[i], origin=start) #apply new length

            shapes.append(shape) #append polygon to list
        
        multi_shape = MultiPolygon(shapes)

        all_points = []
        for poly in multi_shape.geoms:
            all_points.extend(poly.exterior.coords)  #list of all points to calculate max dist between 2 points and use as reference chord length

        points_array = np.array(all_points)

        from scipy.spatial.distance import pdist
        total_chord = np.max(pdist(points_array))  #find max distance

        scale_ratio = 1/total_chord

        multi_shape = scale(multi_shape, xfact=scale_ratio, yfact=scale_ratio, origin=(0, 0))  #normalize chord length to 1

        multi_shape = rotate(multi_shape, angle=-1*np.rad2deg(angle_of_attack), origin=(0.25, 0))

        if reflection:
            multi_shape = scale(multi_shape, xfact=1, yfact=-1, origin=(0,0))
        
        return multi_shape










def dat_list():
    folder_path = Path("../cleaned_foils").resolve()

    unsorted = folder_path.glob("*.dat")
    return sorted(unsorted)
