import os
from .utils import run_gromacs_command, tremd_temperature_ladder, count_residues_in_gro
from importlib.resources import files


def run_em(gmx_path, basename, arg="", debug=False, logger=None):
    """
    Run energy minimization using GROMACS.

    Parameters:
        gmx_path (str): Path to the GROMACS executable (e.g., "gmx").
        basename (str): Base filename (without extension) of the system.
        arg (str): Optional space-separated arguments:
                   [0] = .mdp file path,
                   [1] = suffix for input .gro file (default: ".solv.ions"),
                   [2] = prefix for output files (default: "em"),
                   [3] = additional mdrun arguments (e.g., "-ntmpi 4").
        debug (bool): If True, print commands without executing them.
        logger (logging.Logger): Optional logger for output messages.
    """

    # Set a default placeholder name if no basename is provided
    base = basename if basename else "PLACEHOLDER"
    print(f"Running energy minimization for {base}...")

    # Split optional CLI arguments
    parts = arg.strip().split(maxsplit=3)

    # Set default .mdp file from internal template if not provided
    default_mdp = files("yagwip.templates").joinpath("em.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)

    # Define suffix for input GRO file (e.g., ".solv.ions")
    suffix = parts[1] if len(parts) > 1 else ".solv.ions"

    # Set output file prefix (e.g., em.tpr, em.trr, etc.)
    tprname = parts[2] if len(parts) > 2 else "em"

    # Optional trailing arguments for mdrun (e.g., "-ntmpi 4")
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    # Construct input and output file names
    input_gro = f"{base}{suffix}.gro"
    tpr_file = f"{tprname}.tpr"

    # Construct GROMACS grompp command
    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )

    # Construct GROMACS mdrun command
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

    # If in debug mode, only print commands
    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")

    # Run the commands using the GROMACS runner function
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_nvt(gmx_path, basename, arg="", debug=False, logger=None):
    """
    Run the NVT equilibration step using GROMACS.

    Parameters:
        gmx_path (str): Path to the GROMACS executable (e.g., "gmx").
        basename (str): Base name of the system files (without extension).
        arg (str): Optional CLI-style arguments (space-separated):
                   [0] = path to .mdp file (default: nvt.mdp from templates),
                   [1] = suffix for input .gro file (default: ".em"),
                   [2] = output file prefix (default: "nvt"),
                   [3] = additional flags for mdrun (e.g., "-ntmpi 4").
        debug (bool): If True, print commands instead of executing them.
        logger (logging.Logger): Logger object for logging output if provided.
    """

    # Use placeholder if no basename was provided
    base = basename if basename else "PLACEHOLDER"
    print(f"Running NVT equilibration for {base}...")

    # Parse user input arguments (up to 4 space-separated values)
    parts = arg.strip().split(maxsplit=3)

    # Set default .mdp file if not provided
    default_mdp = files("yagwip.templates").joinpath("nvt.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)

    # Set suffix for the input .gro file (usually .em from the prior energy minimization step)
    suffix = parts[1] if len(parts) > 1 else ".em"

    # Define prefix for the output TPR and trajectory files
    tprname = parts[2] if len(parts) > 2 else "nvt"

    # Optional additional arguments for the mdrun command
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    # Construct input and output file names
    input_gro = f"{base}{suffix}.gro"
    tpr_file = f"{tprname}.tpr"

    # Construct GROMACS grompp command
    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )

    # Construct GROMACS mdrun command
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

    # If in debug mode, only print commands
    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")

    # Run the commands using the GROMACS runner function
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_npt(gmx_path, basename, arg="", debug=False, logger=None):
    """
    Run the NPT equilibration step using GROMACS.

    Parameters:
        gmx_path (str): Path to the GROMACS executable (e.g., "gmx").
        basename (str): Base name for input/output files (e.g., system identifier).
        arg (str): Optional CLI-style arguments (space-separated):
                   [0] = .mdp file (defaults to npt.mdp from templates),
                   [1] = suffix for the input .gro file (default: ".nvt"),
                   [2] = output prefix for TPR and trajectory files (default: "npt"),
                   [3] = additional arguments for mdrun (e.g., "-ntmpi 8").
        debug (bool): If True, commands are printed but not executed.
        logger (Logger): Optional logger to capture output.
    """

    # Use placeholder if no basename was provided
    base = basename if basename else "PLACEHOLDER"
    print(f"Running NPT equilibration for {base}...")

    # Parse user input arguments (up to 4 space-separated values)
    parts = arg.strip().split(maxsplit=3)

    # Set default .mdp file if not provided
    default_mdp = files("yagwip.templates").joinpath("npt.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)

    # Determine suffix of the input .gro file (usually ".nvt" from the prior equilibration step)
    suffix = parts[1] if len(parts) > 1 else ".nvt"
    tprname = parts[2] if len(parts) > 2 else "npt"
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    # Construct input and output file names
    input_gro = f"{base}{suffix}.gro"
    tpr_file = f"{tprname}.tpr"

    # Construct GROMACS grompp command
    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )

    # Construct GROMACS mdrun command
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

    # If in debug mode, only print commands
    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")

    # Run the commands using the GROMACS runner function
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_production(gmx_path, basename, arg="", debug=False, logger=None):
    """
    Run the production molecular dynamics (MD) simulation using GROMACS.

    Parameters:
        gmx_path (str): Path to the GROMACS executable (e.g., "gmx").
        basename (str): Base name for identifying the simulation system.
        arg (str): Optional space-separated user arguments:
                   [0] = .mdp file path (default: production.mdp from templates),
                   [1] = input .gro file prefix (default: "npt."),
                   [2] = output prefix for the production run (default: "md1ns"),
                   [3] = optional mdrun suffix (e.g., "-ntmpi 4 -ntomp 8").
        debug (bool): If True, print commands without executing.
        logger (Logger): Optional logger to capture command output.
    """

    # Use placeholder if no basename was provided
    base = basename if basename else "PLACEHOLDER"
    print(f"Running production MD for {base}...")

    # Parse user input arguments (up to 4 space-separated values)
    parts = arg.strip().split(maxsplit=3)

    # Set default .mdp file if not provided
    default_mdp = files("yagwip.templates").joinpath("production.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
    inputname = parts[1] if len(parts) > 1 else "npt."
    outname = parts[2] if len(parts) > 2 else "md1ns"
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    # Construct input and output file names
    input_gro = f"{inputname}gro"
    tpr_file = f"{outname}.tpr"

    # Construct GROMACS grompp command
    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )

    # Construct GROMACS mdrun command
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {outname} {mdrun_suffix}"

    # If in debug mode, only print commands
    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")

    # Run the commands using the GROMACS runner function
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_tremd(gmx_path, basename, arg="", debug=False):
    """
    Compute a TREMD temperature ladder for replica exchange MD simulations using
    the van der Spoel temperature predictor. Optionally write output to file or print to console.

    Parameters:
        gmx_path (str): Path to the GROMACS executable (not used in this function but kept for interface consistency).
        basename (str): Base name of the system (not used directly here).
        arg (str): Argument string expected to be 'calc <filename.gro>'.
        debug (bool): If True, print the temperature ladder to the console instead of writing to file.
    """

    # Split and validate the command-line-style arguments
    args = arg.strip().split()
    if len(args) != 2 or args[0].lower() != "calc":
        print("Usage: tremd calc <filename.gro>")
        return

    # Resolve the absolute path of the input .gro file
    gro_path = os.path.abspath(args[1])
    if not os.path.isfile(gro_path):
        print(f"[ERROR] File not found: {gro_path}")
        return

    # Parse the .gro file to count the number of protein and water residues
    try:
        protein_residues, water_residues = count_residues_in_gro(gro_path)
        print(f"[INFO] Found {protein_residues} protein residues and {water_residues} water residues.")
    except Exception as e:
        print(f"[ERROR] Failed to parse .gro file: {e}")
        return

    # Prompt user for simulation parameters: Tlow, Thigh, and desired exchange probability
    try:
        Tlow = float(input("Initial Temperature (K): "))
        Thigh = float(input("Final Temperature (K): "))
        Pdes = float(input("Exchange Probability (0 < P < 1): "))
    except ValueError:
        print("[ERROR] Invalid numeric input.")
        return

    # Validate temperature and exchange probability inputs
    if not (0 < Pdes < 1):
        print("[ERROR] Exchange probability must be between 0 and 1.")
        return
    if Thigh <= Tlow:
        print("[ERROR] Final temperature must be greater than initial temperature.")
        return

    # Try to compute the temperature ladder using the van der Spoel algorithm
    try:
        temperatures = tremd_temperature_ladder(
            Tlow=Tlow,
            Thigh=Thigh,
            Pdes=Pdes,
            Nw=water_residues,
            Np=protein_residues,
            Hff=0,      # Assume no hydrogen force field correction
            Vs=0,       # Assume no virtual sites
            PC=1,       # Protein correction mode
            WC=0,       # No water correction
            Tol=0.0005  # Convergence tolerance
        )

        # In debug mode, print the temperature ladder to the console
        if debug:
            print("[DEBUG MODE] TREMD temperature ladder:")
            for i, temp in enumerate(temperatures):
                print(f"Replica {i + 1}: {temp:.2f} K")

        # Otherwise, save it to a file
        else:
            out_file = "TREMD_temp_ranges.txt"
            with open(out_file, 'w') as f:
                f.write("# TREMD Temperature Ladder\n")
                for i, temp in enumerate(temperatures):
                    f.write(f"Replica {i + 1}: {temp:.2f} K\n")
            print(f"[TREMD] Temperature ladder saved to {out_file}")

    # Catch any errors during calculation or file I/O
    except Exception as e:
        print(f"[ERROR] Temperature calculation failed: {e}")
