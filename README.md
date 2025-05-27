# Yet Another GROMACS Wrapper In Python

# 🧪 GROMACS Simulation Automation Library

This Python library automates the setup and execution of GROMACS molecular dynamics (MD) simulations, including support for both standard MD and Replica Exchange Molecular Dynamics (REMD). It provides tools to generate system files, manage directories, chain jobs on HPC clusters, and extract simulation results.

---

## 📦 Modules Overview

### `gromacs_sim.py`

Defines the `GromacsSim` class, which encapsulates all major GROMACS operations for a single simulation. Core features include:

- System setup (`pdb2gmx`, `solvate`, `genion`)
- Energy minimization and equilibration (`em`, `nvt`, `npt`)
- Production MD setup and execution
- Postprocessing (PBC correction, RMSD/RMSF, energy extraction)
- Automated command execution with optional debug mode
- File cleanup and job script creation

**Usage Example:**
```python
sim = GromacsSim("simdir", "ligand", "gmx")
sim.clean_all_except()
sim.pdb2gmx("15\n")
sim.solvate()
sim.genion("13\n")
sim.em()
sim.nvt()
sim.npt()
sim.prepare_run()

