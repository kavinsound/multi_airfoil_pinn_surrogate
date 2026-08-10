import json
from pathlib import Path
import os
import sys

def checkFile(file_name):
    p = Path(file_name)

    data = {
        "status": "not started"
    }

    if not p.exists():
       with open(p, "w") as f:
           json.dump(data, f, indent=4) 


def updateStatus(file_name, status=0):
    p = Path(file_name)

    checkFile(file_name)

    options = ["not started", "meshing", "solving", "completed"]

    if status >= len(options):
        status = 0 #shouldnt happen
        print("invalid status option pls fix")

    data = {
        "status": options[status]
    }

    with open(p, "w") as f:
        json.dump(data, f, indent=4)

    #add print statement here maybe?


if __name__ == "__main__":
    target_json = "caseStatus.json"
    option = int(sys.argv[1])

    updateStatus(target_json, option)
    