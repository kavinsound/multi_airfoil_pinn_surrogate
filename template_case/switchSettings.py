import os
import glob
import shutil
import sys
from pathlib import Path

CASE_DIR = Path(__file__).parent.resolve() / "system"


def switchToPimple():
    settings_folder = CASE_DIR / "pimpleSettings"

    file_list = (CASE_DIR / "simpleSettings").glob("*")

    for file in file_list:
        name = file.stem
        if os.path.exists(CASE_DIR / name):
            os.remove(CASE_DIR / name)

    for file in settings_folder.iterdir():
        shutil.copy2(file, CASE_DIR)

def switchToSimple():
    settings_folder = CASE_DIR / "simpleSettings"

    file_list = (CASE_DIR / "pimpleSettings").glob("*")

    for file in file_list:
        name = file.stem
        if os.path.exists(CASE_DIR / name):
            os.remove(CASE_DIR / name)

    for file in settings_folder.iterdir():
        shutil.copy2(file, CASE_DIR)






if __name__ == "__main__":
    input = sys.argv[1]

    if input == "p":
        switchToPimple()
    elif input == "s":
        switchToSimple()
    