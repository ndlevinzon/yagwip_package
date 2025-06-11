import subprocess
import numpy as np
import math
import logging
import time
import os


def run_gromacs_command(command, pipe_input=None, debug=False, logger=None):
    """
    Executes a shell command for GROMACS, with optional piping and logging.

    Parameters:
        command (str): The shell command to execute.
        pipe_input (str, optional): Optional string input to pipe into the command (e.g., group selection like "13\n").
        debug (bool): If True, prints the command but does not execute it.
        logger (Logger, optional): Optional logger to capture stdout, stderr, and execution info.
    """

    # Log or print the command about to be executed
    if logger:
        logger.info(f"[RUNNING] {command}")
    else:
        print(f"[RUNNING] {command}")

    # In debug mode, do not execute the command
    if debug:
        msg = "[DEBUG MODE] Command not executed."
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        return

    try:
        # Execute the command in a subprocess, piping input if necessary
        result = subprocess.run(
            command,
            input=pipe_input,            # e.g., group index input for genion
            shell=True,
            capture_output=True,         # Capture both stdout and stderr
            text=True                    # Treat input/output as strings
        )

        # Handle non-zero return code (error)
        if result.returncode != 0:
            err_msg = f"[ERROR] Command failed with return code {result.returncode}"
            if logger:
                logger.error(err_msg)
                if result.stderr.strip():
                    logger.error(f"[STDERR] {result.stderr.strip()}")
                if result.stdout.strip():
                    logger.info(f"[STDOUT] {result.stdout.strip()}")
            else:
                print(err_msg)
                print("[STDERR]", result.stderr.strip())
                print("[STDOUT]", result.stdout.strip())
        else:
            # Successful command execution: display or log stdout if present
            if result.stdout.strip():
                if logger:
                    logger.info(f"[STDOUT] {result.stdout.strip()}")
                else:
                    print(result.stdout.strip())

    except Exception as e:
        # Handle unexpected exceptions during subprocess execution
        if logger:
            logger.exception(f"[EXCEPTION] Failed to run command: {e}")
        else:
            print(f"[EXCEPTION] Failed to run command: {e}")


def setup_logger(debug_mode=False):
    """
    Sets up a logger for the YAGWIP application.

    Parameters:
        debug_mode (bool): If True, sets console logging to DEBUG level; otherwise INFO.

    Returns:
        logging.Logger: Configured logger instance for use throughout the CLI.
    """

    # Create or retrieve the logger with a fixed name
    logger = logging.getLogger("yagwip")

    # Set the logger to the most verbose level to ensure all messages are captured
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplicate logs if the function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler Setup ---
    ch = logging.StreamHandler()                                # Handler for stdout/stderr
    ch_level = logging.DEBUG if debug_mode else logging.INFO    # Use DEBUG level in debug mode
    ch.setLevel(ch_level)

    # Define the log message format for console output
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)

    # Attach the console handler to the logger
    logger.addHandler(ch)

    # --- File Handler Setup ---
    # Generate a timestamped filename for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logfile = f"yagwip_{timestamp}.log"

    # Create a file handler to log everything, regardless of debug mode
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)  # Capture all details regardless of debug mode
    fh.setFormatter(formatter)

    # Attach the file handler to the logger
    logger.addHandler(fh)

    # Optional: Notify the user where logs are being written
    if not debug_mode:
        logger.info(f"[LOGGING] Output logged to {logfile}")
    else:
        logger.debug(f"[LOGGING] Debug logging active; also writing to {logfile}")

    # Return the configured logger object
    return logger


def complete_loadpdb(text, line=None, begidx=None, endidx=None):
    """Autocomplete PDB filenames in current directory"""
    if not text:
        completions = [f for f in os.listdir() if f.endswith(".pdb")]
    else:
        completions = [f for f in os.listdir() if f.startswith(text) and f.endswith(".pdb")]
    return completions


def append_ligand_coordinates_to_gro(protein_gro, ligand_pdb, combined_gro):
    coords = []
    with open(ligand_pdb, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))

    with open(protein_gro, 'r') as fin:
        lines = fin.readlines()

    header = lines[:2]
    atom_lines = lines[2:-1]
    box = lines[-1]
    total_atoms = len(atom_lines) + len(coords)

    with open(combined_gro, 'w') as fout:
        fout.write(header[0])
        fout.write(f"{total_atoms}\n")
        fout.writelines(atom_lines)

        for i, (x, y, z) in enumerate(coords, start=1):
            fout.write(f"LIG     LIG  {i:>5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        fout.write(box)

def include_ligand_itp_in_topol(topol_top, ligand_itp, ligand_name="LIG"):
    with open(topol_top, 'r') as f:
        lines = f.readlines()

    with open(topol_top, 'w') as f:
        inserted_include = False
        inserted_mol = False
        for line in lines:
            f.write(line)
            if not inserted_include and '#include' in line and 'forcefield.itp' in line:
                f.write(f'#include "{ligand_itp}"\n')
                inserted_include = True
            if '[ molecules ]' in line:
                inserted_mol = True
            elif inserted_mol and line.strip() == '':
                f.write(f'{ligand_name}    1\n')
                inserted_mol = False


def count_residues_in_gro(gro_path, water_resnames=("SOL",)):
    """
    Parses a GROMACS .gro file to count protein and water residues.

    Parameters:
        gro_path (str): Path to the .gro file.
        water_resnames (tuple): Tuple of residue names considered as water.

    Returns:
        dict: {'protein': int, 'water': int}
    """
    residue_ids = set()
    water_ids = set()

    with open(gro_path, 'r') as f:
        lines = f.readlines()

    # Atom lines are from line 3 to N-2 (last two lines are box vectors)
    for line in lines[2:-1]:
        if len(line) < 20:
            continue  # skip malformed lines

        res_id = int(line[:5].strip())
        res_name = line[5:10].strip()

        if res_name in water_resnames:
            water_ids.add(res_id)
        else:
            residue_ids.add(res_id)

    protein_count = len(residue_ids - water_ids)
    water_count = len(water_ids)

    return protein_count, water_count


# Based on http://dx.doi.org/10.1039/b716554d
def tremd_temperature_ladder(Nw, Np, Tlow, Thigh, Pdes, WC=3, PC=1, Hff=0, Vs=0, Alg=0, Tol=0.001):
    """
    Generate a temperature ladder for temperature replica exchange molecular dynamics (TREMD).

    Parameters:
        Nw (int): Number of water molecules
        Np (int): Number of protein residues
        Tlow (float): Minimum temperature (K)
        Thigh (float): Maximum temperature (K)
        Pdes (float): Desired exchange probability between replicas (0 < P < 1)
        WC (int): Water constraints (3 = all constraints)
        PC (int): Protein constraints (1 = H atoms only, 2 = all, 0 = none)
        Hff (int): Hydrogen force field switch (0 = standard, 1 = different model)
        Vs (int): Include volume correction (1 = yes)
        Alg (int): Not used in this version (reserved for algorithm control)
        Tol (float): Tolerance for exchange probability convergence

    Returns:
        List[float]: Ladder of temperatures suitable for TREMD simulation
    """

    # Empirical coefficients from Patriksson and van der Spoel (2008)
    A0, A1 = -59.2194, 0.07594
    B0, B1 = -22.8396, 0.01347
    D0, D1 = 1.1677, 0.002976
    kB = 0.008314   # Boltzmann constant in kJ/mol/K
    maxiter = 100   # Maximum number of iterations for convergence

    # Estimate number of hydrogen atoms and virtual sites (VC) based on model
    if Hff == 0:
        Nh = round(Np * 0.5134)
        VC = round(1.91 * Nh) if Vs == 1 else 0
        Nprot = Np
    else:
        Npp = round(Np / 0.65957)
        Nh = round(Np * 0.22)
        VC = round(Np + 1.91 * Nh) if Vs == 1 else 0
        Nprot = Npp

    # Degrees of freedom corrections based on constraints
    NC = Nh if PC == 1 else Np if PC == 2 else 0
    Ndf = (9 - WC) * Nw + 3 * Np - NC - VC      # Total degrees of freedom
    FlexEner = 0.5 * kB * (NC + VC + WC * Nw)   # Internal flexibility energy

    # Probability evaluation function for exchange efficiency
    def myeval(m12, s12, CC, u):
        arg = -CC * u - (u - m12) ** 2 / (2 * s12 ** 2)
        return np.exp(arg)

    # Numerical integration using midpoint method for exchange probability contribution
    def myintegral(m12, s12, CC):
        umax = m12 + 5 * s12
        du = umax / 100
        u_vals = np.arange(0, umax, du)
        vals = [myeval(m12, s12, CC, u + du / 2) for u in u_vals]
        pi = np.pi
        return du * sum(vals) / (s12 * np.sqrt(2 * pi))

    # Initialize list of temperatures with the lowest value
    temps = [Tlow]

    # Iteratively compute the next temperature until reaching Thigh
    while temps[-1] < Thigh:
        T1 = temps[-1]                              # Last accepted temperature
        T2 = T1 + 1 if T1 + 1 < Thigh else Thigh    # Initial guess for next temperature
        low, high = T1, Thigh
        iter_count = 0
        piter = 0
        forward = True                              # Flag for adjusting search direction

        # Newton-like iteration to find T2 that yields desired exchange probability
        while abs(Pdes - piter) > Tol and iter_count < maxiter:
            iter_count += 1
            mu12 = (T2 - T1) * ((A1 * Nw) + (B1 * Nprot) - FlexEner)
            CC = (1 / kB) * ((1 / T1) - (1 / T2))
            var = Ndf * (D1 ** 2 * (T1 ** 2 + T2 ** 2) + 2 * D1 * D0 * (T1 + T2) + 2 * D0 ** 2)
            sig12 = np.sqrt(var)

            # Two components of the exchange probability
            I1 = 0.5 * math.erfc(mu12 / (sig12 * np.sqrt(2)))
            I2 = myintegral(mu12, sig12, CC)
            piter = I1 + I2

            # Adjust T2 up or down depending on current probability
            if piter > Pdes:
                if forward:
                    T2 += 1.0
                else:
                    low = T2
                    T2 = low + (high - low) / 2
                if T2 >= Thigh:
                    T2 = Thigh
            else:
                if forward:
                    forward = False
                    low = T2 - 1.0
                high = T2
                T2 = low + (high - low) / 2

        # Append rounded temperature to the list
        temps.append(round(T2, 2))

    print("Please Cite: 'Alexandra Patriksson and David van der Spoel, A temperature predictor for parallel tempering "
          "\n"
          "simulations Phys. Chem. Chem. Phys., 10 pp. 2073-2077 (2008)'")

    return temps


def complete_loadgro(text, line=None, begidx=None, endidx=None):
    """Autocomplete .solv.ions.gro filename in current directory"""
    if not text:
        completions = [f for f in os.listdir() if f.endswith("solv.ions.gro")]
    else:
        completions = [f for f in os.listdir() if f.startswith(text) and f.endswith("solv.ions.gro")]
    return completions


def insert_itp_into_top_files(itp_path_list, root_dir="."):
    """
    Rewrite all topol.top files to include only the provided list of ITP paths,
    removing any existing non-user-specified includes.
    """
    top_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "topol.top":
                top_files.append(os.path.join(root, file))

    for top_file in top_files:
        with open(top_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        inserted = False

        for line in lines:
            # Remove old includes of the form #include "..."
            if line.strip().startswith("#include"):
                continue
            new_lines.append(line)

        # Find insertion point (after forcefield include block)
        insert_idx = 0
        for i, line in enumerate(new_lines):
            if "#include" in line and "forcefield" in line:
                insert_idx = i + 1
                break

        # Inject all custom includes
        include_lines = [f'#include "{path}"\n' for path in itp_path_list]
        new_lines[insert_idx:insert_idx] = include_lines

        # Write updated file
        with open(top_file, 'w') as f:
            f.writelines(new_lines)

        print(f"[UPDATED] Injected {len(itp_path_list)} custom includes into {top_file}")
