import pyvista as pv
from pathlib import Path
import numpy as np

def readCase(path):
    case_path = Path(path)
    mesh = pv.read(case_path / "para.foam")

    print(mesh.keys())





if __name__ == "__main__":
    sample_target = "../sample_case/"

    readCase(sample_target)