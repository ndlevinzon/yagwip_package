# YAGWIP: Yet Another GROMACS Wrapper In Python #

YAGWIP automates the setup and execution of GROMACS molecular dynamics (MD) simulations, including support for both standard MD and Replica Exchange Molecular Dynamics (REMD). It provides tools to generate system files, manage directories, chain jobs on HPC clusters, and extract simulation results. It was originally developed by the talented Dr. Olivier Mailhot.

---

![Alt Text](docs/yagwip1.0_build_sim.png)
 
## Features
- Interactive command loop using Python’s built-in cmd module.
- Dynamic debug mode with toggling and on/off options.
- PDB file loading with automatic setup of the GromacsSim object.
- Simplified pdb2gmx command execution.
- Logging and debug-aware execution of commands.

## CLI Code Structure
### __init__(self, gmx_path)

Initializes the CLI instance with:

    debug: Boolean debug flag

    gmx_path: Path to the GROMACS executable (e.g. "gmx")

    logger: Configured logger

    current_pdb_path: Stores the full path to the loaded PDB

    basename: Base filename for simulation files

    sim: Instance of GromacsSim (simulation manager)

### Methods
init_sim(self)
    Initializes the GromacsSim object after a PDB is loaded:
    Sets working directory and base filename.
    Enables debug mode if active in the CLI.

do_debug(self, arg)
    Toggle or explicitly set debug mode:
        debug         # toggle
        debug on      # set to ON
        debug off     # set to OFF
    Syncs with the internal logger and, if a simulation is initialized, sets its debug flag accordingly.
    
do_loadpdb(self, arg)
    Loads a .pdb file from the current directory:

loadPDB mystructure.pdb
    Sets current_pdb_path.
    Automatically calls init_sim() to prepare the simulation object.

do_pdb2gmx(self, arg)
    Runs pdb2gmx using the loaded PDB file:
        pdb2gmx               # run with defaults
        pdb2gmx -ignh -ff amber99sb  # with extra args
    Requires a previously loaded PDB.
    
do_quit(self, _)
    Exits the CLI:
    quit

default(self, line)
    Handles unknown commands gracefully.
    Running the CLI

## To start the CLI:
```
python yagwip.py
```
You will see:
```
Welcome to YAGWIP. Type help or ? to list commands.
YAGWIP>
```
## Future Development Ideas
Add More GROMACS Commands

Create new do_<command> methods for:

    solvate

    genion

    em, nvt, npt, mdrun

    rms, rmsf, trjconv, energy

Each should wrap the appropriate method in GromacsSim.
 Argument Parsing Improvements

Use argparse or shlex.split() for better CLI argument handling.
 Persistent Session State

Optionally track and save simulation states across sessions.
 Autocompletion

Implement tab-completion support using the cmd module.
 Test Mode

Add a test mode to validate GROMACS setup, environment variables, and PDB compatibility.
Dependencies

    Python 3.6+

    GROMACS installed and accessible via gmx or a module

    Your local GromacsSim class

    logger.py providing setup_logger(debug_mode)

Example: python yagwip.py -f input.pdb
```
# Turn on debug mode for verbose output
debug on

# Load the input PDB structure
loadpdb my_ligand.pdb

# Run PDB2GMX to prepare the topology
pdb2gmx

# Add water box and solvate the system
solvate

# Add ions (Na+, Cl-) and neutralize
genion

# Energy minimization
em

# NVT equilibration
nvt

# NPT equilibration
npt

# Prepare production run
prepare_run

# Run production
production

python yagwip.py -f run_pipeline.groleap
```
