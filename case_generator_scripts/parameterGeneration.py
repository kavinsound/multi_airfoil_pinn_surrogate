import json
import math
import os
from dataclasses import dataclass, asdict
import numpy as np 
from scipy.stats import qmc 


@dataclass
class AirfoilConfig:
    n: int               # Number of active airfoils (1 to 3)
    ids: list            # Fixed length 3
    Re: float            # Reynolds number
    alpha_global: float  # In radians
    chords: list         # Fixed length 3
    deflections: list    # Fixed length 2 (for extra foils)
    gaps: list           # Fixed length 2 vectors [dx, dy] (relative to previous foil TE)


class SobolAirfoilGenerator:
    # Dimensions used per extra airfoil (chord, deflection, gap_r, gap_theta)
    DIMS_PER_EXTRA_AIRFOIL = 4
    MAX_EXTRA_AIRFOILS = 2  # up to 3 airfoils total
    # Dimensions for base state: n_airfoils, Re, alpha_global
    BASE_DIMS = 3  

    def __init__(self, state_file="airfoil_state.json", seed=42):
        self.state_file = state_file
        self.seed = seed
        self.dim = self.BASE_DIMS + self.MAX_EXTRA_AIRFOILS * self.DIMS_PER_EXTRA_AIRFOIL

        # Scrambled=False as per your original script
        self.sampler = qmc.Sobol(d=self.dim, scramble=False, seed=self.seed)

        self.index = 0
        self.configs = []  # Full history of generated configs
        self._load_state()
        if self.index > 0:
            self.sampler.fast_forward(self.index)

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.index = state.get("index", 0)
            self.configs = state.get("configs", [])

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump({"index": self.index, "configs": self.configs}, f, indent=4)

    def _next_sobol(self):
        sample = self.sampler.random(1)[0]
        self.index += 1
        return sample

    def generate(self):
        u = self._next_sobol()

        # 1. Base Parameters
        n = 1 + int(u[0] * 3)  # 1 to 3 airfoils
        
        # Reynolds number: 10^4 (10,000) to 10^6 (1,000,000 - turbulent but incompressible)
        Re = 10 ** (4 + u[1] * 2) 
        
        # Global Angle of Attack in RADIANS (e.g., -10 to +10 degrees converted to rad)
        alpha_global = math.radians(-10 + 20 * u[2])

        # 2. Initialize fixed-size lists (padding out to max capacities)
        chords = [1.0, 0.0, 0.0]
        deflections = [0.0, 0.0]  # Deflection of foil 2, deflection of foil 3
        gaps = [[0.0, 0.0], [0.0, 0.0]]  # Gap vector 1->2, Gap vector 2->3

        # --- INSERT YOUR IDS GENERATION LOGIC HERE ---
        # Example placeholder matching your request for a list of 3
        ids = ["base_id", "none", "none"] 
        # ---------------------------------------------

        # 3. Extract Sobol spaces sequentially for the extra airfoils
        for i in range(1, 3):  # i = 1 (second foil), i = 2 (third foil)
            base_idx = self.BASE_DIMS + (i - 1) * self.DIMS_PER_EXTRA_AIRFOIL
            u_chord, u_defl, u_gap_r, u_gap_theta = u[base_idx:base_idx + 4]
            
            # Populate chord parameters
            chords[i] = 0.3 + 0.7 * u_chord
            
            # Deflection angles for extra foils (in radians, mapping a range like -15 to +15 deg)
            deflections[i-1] = math.radians(-15 + 30 * u_defl)

            # Polar gap translation to Cartesian vectors [dx, dy]
            gap_r = 0.05 + 0.3 * u_gap_r
            gap_theta = u_gap_theta * 2 * math.math.pi
            gaps[i-1] = [gap_r * math.cos(gap_theta), gap_r * math.sin(gap_theta)]

        # 4. Construct complete dataclass object
        cfg = AirfoilConfig(
            n=n,
            ids=ids,
            Re=Re,
            alpha_global=alpha_global,
            chords=chords,
            deflections=deflections,
            gaps=gaps
        )

        self.configs.append(asdict(cfg))
        self._save_state()
        return cfg

    def save_config(self, cfg: AirfoilConfig, filename="airfoil_config.json"):
        with open(filename, "w") as f:
            json.dump(asdict(cfg), f, indent=4)


if __name__ == "__main__":
    gen = SobolAirfoilGenerator(state_file="airfoil_state.json", seed=42)
    for _ in range(5):
        cfg = gen.generate()
        print(f"Active foils (n): {cfg.n}")
        print(f"Alpha (rad): {cfg.alpha_global:.4f} | Re: {cfg.Re:.1f}")
        print(f"Chords: {cfg.chords}")
        print(f"Deflections: {cfg.deflections}")
        print(f"Gaps: {cfg.gaps}\n" + "-"*40)