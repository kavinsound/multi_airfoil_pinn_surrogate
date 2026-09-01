# Multi-Airfoil CFD Dataset Generation & Neural Operator Surrogate

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2112-green.svg)](https://openfoam.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![SLURM](https://img.shields.io/badge/SLURM-HPC-yellow.svg)](https://slurm.schedmd.com/)


**Status:** 🔄 Data Generation Pipeline (Complete) | ⚙️ Preprocessing (Complete) | 🧠 ML Model Architecture (In Progress)

---

## 📖 Abstract

This repository serves as the foundational pipeline for my undergraduate research project focused on accelerating Computational Fluid Dynamics (CFD) simulations using Geometric Deep Learning. The goal is to generate a high-fidelity dataset of 2D turbulent flows over multi-airfoil configurations and develop a Physics-Informed Neural Operator to act as a digital twin for aerodynamic analysis. The entire pipeline is fully automated and designed for high-performance computing environments.

---

## 🔬 Project Pipeline

The workflow is fully automated, spanning from geometric generation to HPC execution and data storage.

### 1️⃣ Parameter Space Exploration
Generates random airfoil configurations and flow conditions using a **Sobol sequence** to ensure a quasi-random, low-discrepancy spread across the design space. The parameters include:

- Reynolds number: $\text{Re} \in [10^4, 10^6]$ (incompressible, transitional regime)
- Angle of attack: $\alpha \in [0^\circ, 45^\circ]$
- Airfoil geometry parameters (NACA profiles, multi-element positions)

2️⃣ Automated Meshing

Constructs a 2D unstructured mesh around the multi-airfoil setup using Gmsh with a Python interface. The algorithm enforces a y+≈1 boundary layer to accurately resolve the near-wall physics for transitional flows.
<p align="center"> <img src="assets/pics/boundary_layer.png" alt="Mesh results showing boundary layer resolution" width="700"/> <br> <em>Figure 1: Unstructured mesh with y+ approx 1 boundary layer resolution around multi-airfoil configuration.</em> </p>

Key meshing features:

    Adaptive refinement near airfoil surfaces

    Wake refinement zone for accurate downstream predictions

    Boundary layer calculation: Δy=y+μρuτ

3️⃣ HPC Simulation

Launches simpleFoam (OpenFOAM) with the transitional k-ω SST model across 1000+ cases on the university SLURM cluster.

    Solver: simpleFoam (steady-state incompressible)

    Turbulence model: k-ω SST with γ-Reθ​ transition model

    Convergence criteria: Residuals < 10−6

4️⃣ Post-processing & Data Storage

Extracts pressure and velocity fields from OpenFOAM results and compresses them into HDF5 files for efficient data loading.

Extracted fields:

    Pressure field: p(x,y)

    Velocity field: u(x,y)=(u,v)

    Surface Coefficients: Pressure Coeff (Cp), Friction Coeff (Cf)

    Lift and drag coefficients: CL​, CD
    
    Turbulence fields: gamma, nut, k, ReThetat, omega​

<p align="center"> <img src="assets/pics/of-U-pic1.png" alt="Velocity field results" width="700"/> <br> <em>Figure 2: Sample velocity field distribution around multi-airfoil configuration with high turbulence.</em> </p>
