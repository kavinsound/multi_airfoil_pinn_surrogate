import glob
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from case_generator_scripts.parameterGeneration import SobolAirfoilGenerator
from case_generator_scripts.stl_generator import mesh_polygon, y_plus_calculator


def generateCase(config, id):
    polygons = mesh_polygon(config)

    coords_list = []

    for polygon in polygons:  # convert polygons to dat format basically
        coords = list(polygon.exterior.coords)
        coords_list.append(coords)

    y_plus_list = y_plus_calculator(config)

    if not os.path.exists("generated_cases"):
        os.mkdir("generated_cases")

    case_path = Path(f"generated_cases/case_{id}")

    shutil.copytree("template_case", case_path)  # copy over template to new path

    Re_n = config.Re

    flow_vel = Re_n * 1.5e-5  # kinematic viscosity of air
    vel_text = f"flowVelocity\t({flow_vel:.3f} 0 0);\n"
    include_file_path = os.path.join(
        case_path, "0.orig", "include", "initialConditions"
    )

    with open(include_file_path, "w") as f:
        f.write(vel_text)

    trisurface_path = os.path.join(case_path, "constant", "triSurface")
    shutil.rmtree(trisurface_path)
    os.mkdir(trisurface_path)

    np.savetxt(os.path.join(trisurface_path, "y_plus"), y_plus_list)
    for i, coords in enumerate(coords_list):
        np.savetxt(os.path.join(trisurface_path, f"airfoil{i + 1}.dat"), coords)

    # all done


def generateBatch(generator, case_list_path, n=64):
    index = generator.index  # read current number of cases
    open(case_list_path, "w").close()  # reset list
    for i in range(n):
        new_config = generator.generate()
        id = index + i
        generateCase(new_config, id)  # hopefully this id is correct
        with open(case_list_path, "a") as f:
            f.write(f"case_{id}")


if __name__ == "__main__":
   generator = SobolAirfoilGenerator(dat_path="cleaned_foils/")

   case_list_path = Path("case_list.txt")
   n = 16 # number of cases

   generateBatch(generator, case_list_path, n)

   #add the job array and stuff later...