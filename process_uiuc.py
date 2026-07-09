from zipfile import ZipFile
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d



# Read one UIUC airfoil file

def load_airfoil(filename):
    points = []

    with open(filename, "r") as f:
        lines = f.readlines()

    # Skip the first line
    for line in lines[1:]:
        values = line.strip().split()

        if len(values) != 2:
            continue

        try:
            x = float(values[0])
            y = float(values[1])
            points.append([x, y])
        except ValueError:
            continue

    return np.array(points)



# Resample airfoil

def resample_airfoil(points, n_points=1000):

    # Distance between neighboring points
    dist = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))


    # some UIUC files have duplicate consecutive points
    keep = np.concatenate(([True], dist > 1e-10))
    points = points[keep]
    dist = dist[keep[1:]]

    # Cumulative distance along the airfoil
    s = np.concatenate(([0], np.cumsum(dist)))

    # Normalize to [0,1]
    s = s / s[-1]

    # Interpolate x and y separately
    fx = interp1d(s, points[:, 0], kind="cubic")
    fy = interp1d(s, points[:, 1], kind="cubic")

    s_new = np.linspace(0, 1, n_points)

    x_new = fx(s_new)
    y_new = fy(s_new)

    return np.column_stack((x_new, y_new))



# Save a processed airfoil

def save_airfoil(points, filename, name):

    with open(filename, "w") as f:

        f.write(name + "\n")

        for x, y in points:
            f.write(f"{x:.8f} {y:.8f}\n")



def main():

    zip_file = "src/coord_seligFmt.zip"

    extract_folder = Path("uiuc_airfoils")
    output_folder = Path("processed_airfoils")

    output_folder.mkdir(exist_ok=True)

    print("Extracting database...")

    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    input_folder = extract_folder / "coord_seligFmt"

    airfoil_files = sorted(input_folder.glob("*.dat"))

    print(f"Found {len(airfoil_files)} airfoils.\n")

    for i, file in enumerate(airfoil_files):

        points = load_airfoil(file)

        # Skip bad files
        if len(points) < 5:
            continue

        new_points = resample_airfoil(points, 1000)

        save_airfoil(
            new_points,
            output_folder / file.name,
            file.stem,
        )

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(airfoil_files)}")

    print(f"Processed airfoils saved to '{output_folder}'")


if __name__ == "__main__":
    main()