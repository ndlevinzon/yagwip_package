# YAGWIP: Yet Another GROMACS Wrapper In Python

Because YAGWIP is written entirely in Python3 with minimal dependencies, we have structured our code to be as developer-friendly as possible. The following outlines our directory structures, as well as how our code is written and utilized. The idea here is to be as extendable as possible, so that others can make whatever additions they need to perform their GROMACS molecular simulations

---

## Project Directory Structure
```
yagwip_package/
├── src/
    ├── ligands/
    ├── jobs/
    ├── md_runs/
├── docs/

```

## Modules Overview

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
from yagwip.gromacs_sim import GromacsSim

sim = GromacsSim("simdir", "ligand", "gmx")
sim.clean_all_except()
sim.pdb2gmx("15\n")
sim.solvate()
sim.genion("13\n")
sim.em()
sim.nvt()
sim.npt()
sim.prepare_run()
```

### 'experiment.py'

Defines the 'Experiment' class, which organizes and initializes multiple replicas of a simulation using the same inputs (PDBs, force fields, MDPs). It automates:

- Directory creation and file setup
- Copying base simulation files to all replicas
- Creation of setup and mdrun job scripts (with chaining)
- Launchable bash starter scripts for SLURM environments

**Usage Example:**
```python
exp = Experiment(
    pdb_file_list=["ligA.pdb", "ligB.pdb"],
    maindir="md_runs",
    basefilesdir="templates",
    jobdir="jobs",
    n_replicas=3,
    ffcode="15\n",
    solcode="13\n",
    water="spce"
)
exp.initialize_dirs_copy_basefiles()
exp.create_all_scripts()
```
### 'remd.py'

Defines the Remd class, which automates Replica Exchange Molecular Dynamics across a temperature range.

Key features:

- Initializes a GROMACSSim for each temperature
- Copies and modifies .mdp files for each replica with specific ref_t
- Runs a standard equilibration pipeline (pdb2gmx → solvate → genion → em → nvt → npt → prepare_run)

Usage Example:
```python
temps = [280, 290, 300, 310, 320]
remd = Remd(
    base_name="vp35",
    main_dir="remd_output",
    base_dirname="vp35",
    temperature_list=temps,
    basefiles_dir="templates",
    gmx_path="gmx"
)
```
### Utilities

Also included are helper functions for:
- Writing SLURM job headers and bash scripts
- Managing file paths and shell command arguments
- Debug mode for dry-run execution

## Quick Start

1. Prepare your .pdb, .mdp, and .ff files
2. Define your experiment or REMD run in Python
3. Run setup scripts or launch SLURM jobs via generated bash scripts

### Folder Structure Example
```
project/
├── templates/              # Contains .mdp and .ff files
├── ligands/                # Input .pdb files
├── jobs/                   # Output job scripts
├── md_runs/                # Output simulation directories
├── run_experiment.py       # Your driver script using Experiment or Remd
```
## Dependencies
- Python 3.x
- GROMACS installed and available on PATH
- SLURM scheduler (for job scripts)
- Bash

