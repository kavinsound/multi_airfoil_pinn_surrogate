import glob  
import subprocess
import os
import pprint
import shutil
from pathlib import Path

import numpy as np
import sqlite3

from case_generator_scripts.parameterGeneration import SobolAirfoilGenerator
from case_generator_scripts.stl_generator import mesh_polygon, y_plus_calculator


def generateCase(config, id):
    file_list = Path("cleaned_foils").glob("*.dat")
    file_list = sorted(file_list)
    polygons = mesh_polygon(config, file_list)

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
      #   pprint.pprint(new_config)
        id = index + i
        generateCase(new_config, id)  # hopefully this id is correct

        with open(case_list_path, "a") as f:
            f.write(f"case_{id}\n")

        initializeSQLITE(id, "job_status.db")
        print(f"Generated case_{id}...")

def initializeSQLITE(id, sql_path):
    conn = sqlite3.connect(sql_path)

    cursor = conn.cursor()

    cursor.execute(
    """
        CREATE TABLE IF NOT EXISTS job_stages (
            job_id TEXT PRIMARY KEY,
            stage INTEGER
        )
    """
    )

    cursor.execute(
    """
        INSERT OR IGNORE INTO job_stages (job_id, stage) 
        VALUES (?, ?)
    """,
        (f"case_{id}", 0),
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
   generator = SobolAirfoilGenerator(dat_path="./cleaned_foils/")
   # print(generator.foil_list)
   case_list_path = Path("case_list.txt")
   n = 16 # number of cases

   generateBatch(generator, case_list_path, n)

   cmd = ["bash", "submit.sh"]
   subprocess.run( #run submit.sh to start the job array
    cmd,
    check=True
   )
