from pathlib import Path
import glob
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, scale
import trimesh
import matplotlib.pyplot as plt
from scipy.special import ndtri


from parameterGeneration import SobolAirfoilGenerator, AirfoilConfig


#                length of id array is number of airfoils (1-3)
def mesh_polygon(
    config
):
    n, ids, angle_of_attack, lengths, deflection_angles, gap_vectors, reflection = config.n, config.ids, config.alpha_global, config.chords, config.deflections, config.gaps, config.reflection
    gap_vectors = np.array(gap_vectors).astype(float)
    reflection = 0
    angle_of_attack = 0  #REMOVE LATER
    file_list = dat_list()  # list of foils
    # print(ids)
    if n == 1:  # only one airfoil is simple
        foil_coords = np.loadtxt(file_list[ids[0]])
        shape = Polygon(foil_coords)

        shape = rotate(
            shape, angle=-1 * np.rad2deg(angle_of_attack), origin=(0.25, 0)
        )  # rotate by attack angle

        if reflection:
            shape = scale(
                shape, xfact=1, yfact=-1, origin=(0, 0)
            )  # reflect for downward

        return [shape]

    else:
        shapes = []  # list to hold each polygon to combine after
        end_point = np.array([1, 0]).astype(float)  # end of each polygon as we add to list
        leading = np.array([0, 0])
        for i in range(n):
            foil_coords = np.loadtxt(file_list[ids[i]])
            
            if i != 0:
                foil_coords += (
                    end_point + gap_vectors[i - 1]
                )  # after first airfoil, calculate starting point by adding the vectors to origin
                leading = end_point + gap_vectors[i-1] #start of airfoil
                end_point = leading + [lengths[i]*np.cos(deflection_angles[i-1]), -1 * lengths[i]*np.sin(deflection_angles[i-1])]
                # calculate new endpoint location to use in next loop
            shape = Polygon(foil_coords)

              # calculate absolute rotation center with new endpoint

            if i != 0:
                shape = rotate(shape, angle=-1 * np.rad2deg(deflection_angles[i-1]), origin=tuple(leading))  # apply individual deflection angle

              # find tip of airfoil to apply scale without moving it around
            shape = scale(shape, xfact=lengths[i], yfact=lengths[i], origin=tuple(leading))  # apply new length

            shapes.append(shape)  # append polygon to list


        multi_shape = MultiPolygon(shapes)

        all_points = []
        for poly in multi_shape.geoms:
            all_points.extend(
                poly.exterior.coords
            )  # list of all points to calculate max dist between 2 points and use as reference chord length

        points_array = np.array(all_points)

        from scipy.spatial.distance import pdist

        total_chord = np.max(pdist(points_array))  # find max distance

        scale_ratio = 1 / total_chord

        multi_shape = scale(multi_shape, xfact=scale_ratio, yfact=scale_ratio, origin=(0, 0))  # normalize chord length to 1

        multi_shape = rotate(multi_shape, angle=-1 * np.rad2deg(angle_of_attack), origin=(0.25, 0))

        if reflection:
            multi_shape = scale(multi_shape, xfact=1, yfact=-1, origin=(0, 0))

        return list(multi_shape.geoms)


def dat_list():
    folder_path = Path("../cleaned_foils").resolve()

    unsorted = folder_path.glob("*.dat")
    return sorted(unsorted)

class InteractiveVisualizer:  #Graph the generated setups to visually confirm quality
    def __init__(self, generator):
        self.generator = generator
        
        # Setup the plot canvas
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.fig.canvas.manager.set_window_title("Airfoil Polygon Inspector")
        
        # Bind the key press controller event
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        
        # Plot the first initial configuration state automatically
        self.draw_next_configuration()

    def draw_next_configuration(self):
        # 1. Generate fresh parameters
        cfg = self.generator.generate()
        
        print(f"Active foils (n): {cfg.n}")
        print(f"Alpha (rad): {cfg.alpha_global:.4f} | Re: {cfg.Re:.1f}")
        print(f"Chords: {cfg.chords}")
        print(f"Deflections: {cfg.deflections}")
        print(f"Gaps: {cfg.gaps}\n" + "-"*40)

        # 2. Clear out the previous frame completely
        self.ax.clear()
        
        # 3. Generate your coordinate arrays using your custom geometry builder
        polygons = mesh_polygon(cfg)
        
        # 4. Render each polygon loop layer to the axis
        for idx, poly in enumerate(polygons):
            poly_arr = np.array(poly.exterior.coords)
            # Separate out X and Y columns
            x, y = poly_arr[:, 0], poly_arr[:, 1]
            
            # Draw line and shaded fill region
            self.ax.plot(x, y, label=f"Element {idx+1}", linewidth=1.5)
            self.ax.fill(x, y, alpha=0.2)
            
        # 5. Presentation formatting
        self.ax.set_title(f"Config #{self.generator.index} (n={cfg.n}) | PRESS [SPACE] FOR NEXT", fontsize=11, fontweight="bold")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.legend(loc="upper right")
        
        # CRITICAL: Keep aspect ratio 1:1 so angles and gaps aren't stretched/distorted
        self.ax.set_aspect("equal", adjustable="box")
        
        # Dynamically fit margins cleanly around the elements
        self.ax.autoscale_view()
        
        # Flush the updates onto the visible window GUI
        self.fig.canvas.draw()

    def on_key(self, event):
        # Listen explicitly for spacebar interactions
        if event.key == " ":
            self.draw_next_configuration()
        # Allow structural escape commands
        elif event.key == "escape":
            plt.close(self.fig)

def extrude_stls(polygon_list):
    n = len(polygon_list)
    stl_list = []
    for i in range(n):
        polygon_mesh = trimesh.creation.extrude_polygon(polygon_list[i], height=1.0)
        stl_list.append(polygon_mesh)
    return stl_list

def y_plus_calculator(config):
    re = config.Re
    nu = 1.5e-5
    layers = []
    for i in range(config.n):
        chord = config.chords[i]

        U = (re * nu) / chord
        
        # 2. Calculate Skin Friction Coefficient (Cf)
        # Using the 1/5th power law correlation
        cf = 0.058 * (re**-0.2)
        
        # 3. Calculate friction velocity (u_tau)
        # u_tau = U * sqrt(Cf / 2)
        u_tau = U * (cf / 2)**0.5
        
        # 4. First cell height
        first_layer = (0.8 * nu) / u_tau
        layers.append(first_layer)
    
    return layers




if __name__ == "__main__":
    gen = SobolAirfoilGenerator(state_file="airfoil_state.json", seed=42)
    
    # Fire up the interactive visual inspection application
    visualizer = InteractiveVisualizer(gen)
    plt.show()