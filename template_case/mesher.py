import glob
import os
import sys
from pathlib import Path
import subprocess

import numpy as np

case_dir = Path(__file__).resolve().parent
helper_dir = case_dir.parent.parent / "case_generator_scripts"
sys.path.append(str(helper_dir))

from case_generator_scripts.gmsh_mesher import createMesh


def meshCase():
    trisurface_dir = os.path.join(case_dir, "constant", "triSurface")

    dat_files = trisurface_dir.glob("*.dat")
    dat_files = sorted(dat_files)

    n = len(dat_files)
    airfoils = []

    for file in dat_files:
        coords = np.loadtxt(file, dtype=np.float64)
        airfoils.append(coords)

    y_plus = np.loadtxt(os.path.join(trisurface_dir, "y_plus"), dtype=np.float64)

    createMesh(airfoils, y_plus, case_dir)




if __name__ == "__main__":
    meshCase()

    meshCheckCMD = ["checkMesh", "-allGeometry", "-allTopology"]
    logfile = "meshCheck.log"
    result = subprocess.run(
        meshCheckCMD,
        check=False,
        stdout=logfile,
        stderr=logfile
    )
