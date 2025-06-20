import subprocess
import numpy as np
import math
import logging
import time
import os
import re


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

    # In debug mode, skip execution and only log/print a message
    if debug:
        msg = "[DEBUG MODE] Command not executed."
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        return

    try:
        # Execute the command with optional piped input
        result = subprocess.run(
            command,
            input=pipe_input,       # Piped input to stdin, if any (e.g., group number)
            shell=True,             # Run command through shell
            capture_output=True,    # Capture both stdout and stderr
            text=True               # Decode outputs as strings instead of bytes
        )

        # Strip leading/trailing whitespace from stderr and stdout
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        error_text = f"{stderr}\n{stdout}".lower()  # Combined output used for keyword-based error checks

        # Check if the command failed based on return code
        if result.returncode != 0:
            err_msg = f"[!] Command failed with return code {result.returncode}"

            # Log or print error details
            if logger:
                logger.error(err_msg)
                if stderr:
                    logger.error(f"[STDERR] {stderr}")
                if stdout:
                    logger.info(f"[STDOUT] {stdout}")
            else:
                print(err_msg)
                if stderr:
                    print("[STDERR]", stderr)
                if stdout:
                    print("[STDOUT]", stdout)

            # Catch atom number mismatch error
            if "number of coordinates in coordinate file" in error_text:
                specific_msg = "[!] Check ligand and protonation: .gro and .top files have different atom counts."
                if logger:
                    logger.warning(specific_msg)
                else:
                    print(specific_msg)

            # Catch periodic improper dihedral type error
            elif "no default periodic improper dih. types" in error_text:
                match = re.search(r'\[file topol\.top, line (\d+)\]', stderr, re.IGNORECASE)
                if match:
                    line_num = int(match.group(1))
                    top_path = "./topol.top"

                    try:
                        with open(top_path, 'r') as f:
                            lines = f.readlines()

                        if 0 <= line_num - 1 < len(lines):
                            if not lines[line_num - 1].strip().startswith(';'):
                                lines[line_num - 1] = f";{lines[line_num - 1]}"
                                with open(top_path, 'w') as f:
                                    f.writelines(lines)

                                msg = (f"[#] Detected improper dihedral error, likely an artifact from AMBER force fields.\n"
                                       f"[#] Commenting out line {line_num} in topol.top and rerunning...")
                                if logger:
                                    logger.warning(msg)
                                else:
                                    print(msg)

                            # Retry the command
                                retry_msg = f"[#] Rerunning command after modifying topol.top..."
                                if logger:
                                    logger.info(retry_msg)
                                else:
                                    print(retry_msg)

                                # Important: recursive retry, but prevent infinite loops
                                return run_gromacs_command(command, pipe_input=pipe_input, debug=debug, logger=logger)

                    except Exception as e:
                        fail_msg = f"[!] Failed to modify topol.top: {e}"
                        if logger:
                            logger.error(fail_msg)
                        else:
                            print(fail_msg)
                else:
                    fallback_msg = "[!] Detected dihedral error, but couldn't find line number in topol.top."
                    if logger:
                        logger.warning(fallback_msg)
                    else:
                        print(fallback_msg)

            return False  # Final return if not resolved

        else:
            # If successful, optionally print/log stdout
            if stdout:
                if logger:
                    logger.info(f"[STDOUT] {stdout}")
                else:
                    print(stdout)
            return True

    except Exception as e:
        # Catch and log any unexpected runtime exceptions (e.g., permission issues)
        if logger:
            logger.exception(f"[!] Failed to run command: {e}")
        else:
            print(f"[!] Failed to run command: {e}")
        return False


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
        logger.info(f"[#] Output logged to {logfile}")
    else:
        logger.debug(f"[#] Debug logging active; also writing to {logfile}")

    # Return the configured logger object
    return logger


def complete_filename(text, suffix, line=None, begidx=None, endidx=None):
    """
    Generic TAB Autocomplete for filenames in the current directory matching a suffix.

    Parameters:
        text (str): The current input text to match.
        suffix (str): The file suffix or pattern to match (e.g., ".pdb", "solv.ions.gro").
    """
    if not text:
        return [f for f in os.listdir() if f.endswith(suffix)]
    else:
        return [f for f in os.listdir() if f.startswith(text) and f.endswith(suffix)]


def append_ligand_atomtypes_to_forcefield(ligand_itp='ligand.itp', ffnonbonded_itp='./amber14sb.ff/ffnonbonded.itp'):
    """
    Appends the [ atomtypes ] section from the ligand.itp file to the forcefield's ffnonbonded.itp file.
    This fixes the improper dihedrals term that AMBER ANTECHAMBER generates for ligands.
    Ensures that this block is only added once by checking for a ";ligand" tag.

    Parameters:
        ligand_itp (str): Path to the ligand .itp file.
        ffnonbonded_itp (str): Path to the forcefield nonbonded types file (typically ffnonbonded.itp).
    """
    # Check if the ligand.itp file exists
    if not os.path.isfile(ligand_itp):
        print(f"[!] ligand.itp not found at {ligand_itp}")
        return

    # Read all lines from ligand.itp
    with open(ligand_itp, 'r') as f:
        lines = f.readlines()

    atomtypes_block = []        # Buffer for collecting atomtypes lines
    inside_atomtypes = False    # Flag to track whether we are inside the [ atomtypes ] section

    # Extract the [ atomtypes ] section before [ moleculetype ]
    for line in lines:
        if line.strip().startswith("[ moleculetype ]"):
            break               # Stop parsing if we reach the [ moleculetype ] section
        if line.strip().startswith("[ atomtypes ]"):
            inside_atomtypes = True
            continue
        if inside_atomtypes:
            atomtypes_block.append(line)

    # If no atomtypes were found, exit with a warning
    if not atomtypes_block:
        print("[#] No [ atomtypes ] section found in ligand.itp. Skipping...")
        return

    # Ensure the ffnonbonded.itp file exists
    if not os.path.isfile(ffnonbonded_itp):
        print(f"[!] ffnonbonded.itp not found at {ffnonbonded_itp}")
        return

    # Prevent duplicate addition by checking for the ";ligand" marker
    with open(ffnonbonded_itp, 'r') as fcheck:
        if ";ligand" in fcheck.read():
            print("[#] ligand section already exists in ffnonbonded.itp. Skipping...")
            return

    # Append the atomtypes block to the end of ffnonbonded.itp
    with open(ffnonbonded_itp, 'a') as fout:
        fout.write("\n;ligand\n")            # Marker for later identification
        fout.writelines(atomtypes_block)     # Write the atomtypes lines

    print(f"[#] Atomtypes from {ligand_itp} appended to {ffnonbonded_itp}.")


def modify_improper_dihedrals_in_ligand_itp(filename='ligand.itp'):
    """
    Converts improper dihedrals from AMBER format to GROMACS format in the [ dihedrals ] section.
    Specifically:
        - Converts func=4 (used in AMBER forcefields for improper torsions) to func=2 (used in GROMACS).
        - Removes the eighth column (periodicity, pn) which is unnecessary for func=2.

    Parameters:
        filename (str): Path to the ligand .itp file (default: 'ligand.itp').
    """
    # Check if the ligand.itp file exists
    if not os.path.isfile(filename):
        print(f"[!] {filename} not found.")
        return

    # Read all lines from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    output_lines = []       # Buffer to store modified lines
    in_dihedrals = False    # Flag to track whether we're inside [ dihedrals ] section
    modified = False        # Flag to report if any line was changed

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect beginning of [ dihedrals ] section (specifically for impropers)
        if stripped.lower().startswith("[ dihedrals ]") and "impropers" in stripped:
            in_dihedrals = True
            output_lines.append(line)   # Keep the section header
            continue

        # If a new section starts, we are no longer inside [ dihedrals ]
        if in_dihedrals and stripped.startswith("[") and "]" in stripped:
            in_dihedrals = False

        # Modify only lines within [ dihedrals ] and skip comments and blank lines
        if in_dihedrals and stripped and not stripped.startswith(";"):
            parts = line.split(";")                              # Split potential comment from the line
            comment = f";{parts[1]}" if len(parts) > 1 else ""   # Preserve any comment
            tokens = parts[0].split()                            # Split numeric fields

            # Check if line looks like an Amber-style improper dihedral
            if len(tokens) >= 8 and tokens[4] == "4":
                tokens[4] = "2"       # Change func from 4 to 2
                del tokens[7]         # Remove the periodicity column (pn)
                new_line = "   " + "   ".join(tokens) + "   " + comment + "\n"
                output_lines.append(new_line)
                modified = True
            else:
                output_lines.append(line)   # Keep the line unchanged if it doesn't match the expected format
        else:
            output_lines.append(line)       # Outside [ dihedrals ] or irrelevant lines

    # If nothing was modified, notify and exit
    if not modified:
        print("[#] No impropers with func=4 found to modify. Skipping...")
        return

    # Write modified lines back to the original file
    with open(filename, 'w') as f:
        f.writelines(output_lines)

    print(f"[#] Improper dihedrals converted to func=2 in {filename}.")


def rename_residue_in_itp_atoms_section(filename='./ligand.itp', old_resname="MOL", new_resname="LIG"):
    """
    Replaces the residue name in the [ atoms ] section and the molecule name in the [ moleculetype ] section
    of a GROMACS .itp file. This is useful for standardizing naming (e.g., replacing "MOL" with "LIG").
    The function performs in-place modification of the file.

    Parameters:
        filename (str): Path to the ligand .itp file (default: './ligand.itp').
        old_resname (str): Residue name to search for (e.g., 'MOL').
        new_resname (str): Replacement residue name (e.g., 'LIG').
    """
    # Read all lines from the input .itp file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize state flags and output list
    in_atoms_section = False
    in_moleculetype_section = False
    modified_lines = []

    # Iterate through each line in the file
    for idx, line in enumerate(lines):
        stripped = line.strip()

        # Detect start of [ atoms ] section
        if stripped.startswith("[ atoms ]"):
            in_atoms_section = True
            modified_lines.append(line)
            continue

        # Detect start of [ moleculetype ] section
        if stripped.startswith("[ moleculetype ]"):
            in_moleculetype_section = True
            modified_lines.append(line)
            continue

        # Inside [ atoms ] section: look for data lines to modify residue name
        if in_atoms_section:
            # Exit section if a blank line or new header is encountered
            if stripped == "" or stripped.startswith("["):
                in_atoms_section = False
                modified_lines.append(line)
                continue
            # Process only non-comment lines
            if not stripped.startswith(";"):
                # Split the line using regex to preserve spacing
                parts = re.split(r'(\s+)', line)

                # Make sure there's a residue field to edit (column 4 = index 6 in split-by-whitespace+space list)
                if len(parts) >= 9 and parts[6].strip() == old_resname:
                    parts[6] = new_resname      # Replace residue name with new one
                line = ''.join(parts)           # Reconstruct the line with original spacing

            modified_lines.append(line)         # Store modified or unmodified line
            continue

        # Inside [ moleculetype ] section: replace molecule name
        if in_moleculetype_section:
            # Preserve empty lines and comments
            if stripped == "" or stripped.startswith(";"):
                modified_lines.append(line)
                continue
            else:
                # Replace the entire first column with LIG and preserve rest of line (e.g., nrexcl)
                tokens = line.split()
                if len(tokens) >= 2:
                    tokens[0] = new_resname
                    line = f"{tokens[0]:<20}{tokens[1]}\n"      # Align first column to 20 chars for formatting
                else:
                    line = f"{new_resname}\n"                   # If only one token present, just replace it

                # Append modified line and exit section
                modified_lines.append(line)
                in_moleculetype_section = False
                continue

        # If not in a special section, leave line unmodified
        modified_lines.append(line)

    # Write modified content back to the same file
    with open(filename, 'w') as f:
        f.writelines(modified_lines)

    print(f"[#] Updated [ atoms ] and [ moleculetype ] sections in {filename}: {old_resname} -> {new_resname}")


def append_ligand_coordinates_to_gro(protein_gro, ligand_pdb, combined_gro="complex.gro"):
    """
    Appends ligand atomic coordinates from a PDB file to an existing GROMACS .gro file
    containing a protein, and writes the combined structure to a new .gro file.

    Parameters:
        protein_gro (str): Path to the input .gro file containing the protein structure.
        ligand_pdb (str): Path to the ligand structure in PDB format.
        combined_gro (str): Output path for the combined protein-ligand .gro file (default: 'complex.gro').
    """

    coords = []

    # Parse coordinates from ligand .PDB
    with open(ligand_pdb, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                res_id = int(line[23:26].strip())       # Residue number
                atom_name = line[13:16].strip()         # Atom name (e.g., CA, HN)
                res_name = line[17:20].strip()          # Residue name (e.g., LIG)
                atom_index = int(line[6:11].strip())    # Atom serial number
                x = float(line[30:38])                  # X Coordinate (A)
                y = float(line[38:46])                  # Y Coordinate (A)
                z = float(line[46:54])                  # Z Coordinate (A)

                # Store parsed ligand atom as a tuple
                coords.append((res_id, res_name, atom_name, atom_index, x, y, z))

    # Read protein .GRO
    with open(protein_gro, 'r') as fin:
        lines = fin.readlines()

    header = lines[:2]                  # First two lines: title and atom count
    atom_lines = lines[2:-1]            # Middle lines: atom entries
    box = lines[-1]                     # Last line: box dimensions

    # Update total atom count (protein atoms + ligand atoms)
    total_atoms = len(atom_lines) + len(coords)

    # Write combined .GRO file
    with open(combined_gro, 'w') as fout:
        fout.write(header[0])           # Title line
        fout.write(f"{total_atoms}\n")  # Updated atom count
        fout.writelines(atom_lines)     # Original protein atoms

        # Write ligand atoms, converting from Angstrom to nm for GROMACS format
        for res_id, res_name, atom_name, atom_index, x, y, z in coords:
            fout.write(f"{res_id:5d}{res_name:<5}{atom_name:>5}{atom_index:5d}{x/10:8.3f}{y/10:8.3f}{z/10:8.3f}\n")

        fout.write(box)                 # Box dimensions, unchanged (last line)

    print(f"[#] Wrote {total_atoms} atoms to {combined_gro}")


def include_ligand_itp_in_topol(topol_top, ligand_itp="./ligand.itp", ligand_name="LIG"):
    """
    Modifies the topology file (topol.top) to include the ligand's .itp file and add an entry for the ligand in
    the [ molecules ] section.

    Parameters:
        topol_top (str): Path to the topology file to modify (e.g., topol.top).
        ligand_itp (str): Path to the ligand's .itp file to include (default: ./ligand.itp).
        ligand_name (str): Residue name of the ligand to add in the [ molecules ] section (default: "LIG").
    """
    # Read the original topology file
    with open(topol_top, 'r') as f:
        lines = f.readlines()

    new_lines = []                  # Holds the modified lines to write back
    inserted_include = False        # Track whether we've inserted the #include line for ligand.itp
    inserted_mol = False            # Track whether we've inserted the ligand entry in [ molecules ]
    in_molecules_section = False    # Track whether we are currently in the [ molecules ] section
    molecules_lines = []            # Buffer for lines in the [ molecules ] section

    for line in lines:
        stripped = line.strip()

        # If we find the forcefield include line, insert ligand.itp include right after it
        if not inserted_include and '#include' in stripped and 'forcefield.itp' in stripped:
            new_lines.append(line)
            new_lines.append(f'#include "{ligand_itp}"\n')  # Insert ligand .itp include
            inserted_include = True
            continue                                        # Skip to next line

        # Detect start of the [ molecules ] section, start buffering lines
        if '[ molecules ]' in stripped:
            in_molecules_section = True
            molecules_lines.append(line)
            continue

        # If we reach the end of the [ molecules ] section or a new section starts
        if in_molecules_section:
            # If we reach the end of section (blank line or new section)
            if stripped == '' or stripped.startswith('['):
                # If ligand wasn't already added, append it now
                if not inserted_mol:
                    molecules_lines.append(f'{ligand_name}    1\n')
                    inserted_mol = True
                # Append buffered section to output
                new_lines.extend(molecules_lines)
                molecules_lines = []                # Clear buffer
                in_molecules_section = False        # Exit section
                new_lines.append(line)              # Append the current section header or blank line
            else:
                # Only add lines that are not already the ligand (avoid duplication)
                if ligand_name not in stripped:
                    molecules_lines.append(line)
        else:
            # Outside [ molecules ] section, append lines as-is
            new_lines.append(line)

    # Handle case where [ molecules ] is at the very end of the file
    if in_molecules_section and not inserted_mol:
        molecules_lines.append(f'{ligand_name}                 1\n')
        new_lines.extend(molecules_lines)

    # Write the modified content back to the topology file
    with open(topol_top, 'w') as f:
        f.writelines(new_lines)


def count_residues_in_gro(gro_path, water_resnames=("SOL",)):
    """
    Parses a GROMACS .gro file to count protein and water residues.
    Used for generating T-REMD Temperature Ladder.

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
    Generate a temperature ladder for temperature replica exchange molecular dynamics (T-REMD).

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


def insert_itp_into_top_files(itp_path_list, root_dir="."):
    """
    Rewrite all topol.top files to include only the provided list of ITP paths,
    removing any existing non-user-specified includes. Used for the SOURCE command.
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

        print(f"[#] Injected {len(itp_path_list)} custom includes into {top_file}")
