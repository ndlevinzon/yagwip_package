"""
utils.py -- YAGWIP Utility Functions
"""

# === Standard Library Imports ===
import os
import re
import subprocess

# === Third-Party Imports ===
import numpy as np

# === Local Imports ===
from .log_utils import auto_monitor


@auto_monitor
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
        logger.info("[RUNNING] %s", command)
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
            input=pipe_input,  # Piped input to stdin, if any (e.g., group number)
            shell=True,  # Run command through shell
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Decode outputs as strings instead of bytes
            check=False,  # Raise an error if the command fails
        )

        # Strip leading/trailing whitespace from stderr and stdout
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        error_text = (
            f"{stderr}\n{stdout}".lower()
        )  # Combined output used for keyword-based error checks

        # Check if the command failed based on return code
        if result.returncode != 0:
            err_msg = f"[!] Command failed with return code {result.returncode}"

            # Log or print error details
            if logger:
                logger.error(err_msg)
                if stderr:
                    logger.error("[STDERR] %s", stderr)
                if stdout:
                    logger.info("[STDOUT] %s", stdout)
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
                match = re.search(
                    r"\[file topol\.top, line (\d+)\]", stderr, re.IGNORECASE
                )
                if match:
                    line_num = int(match.group(1))
                    top_path = "./topol.top"

                    try:
                        with open(top_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        if 0 <= line_num - 1 < len(lines):
                            if not lines[line_num - 1].strip().startswith(";"):
                                lines[line_num - 1] = f";{lines[line_num - 1]}"
                                with open(top_path, "w", encoding="utf-8") as f:
                                    f.writelines(lines)

                                msg = (
                                    f"[#] Detected improper dihedral error, likely an artifact from AMBER forcefields."
                                    f" Commenting out line {line_num} in topol.top and rerunning..."
                                )
                                if logger:
                                    logger.warning(msg)
                                else:
                                    print(msg)

                                # Retry the command
                                retry_msg = (
                                    "[#] Rerunning command after modifying topol.top..."
                                )
                                if logger:
                                    logger.info(retry_msg)
                                else:
                                    print(retry_msg)

                                # Important: recursive retry, but prevent infinite loops
                                return run_gromacs_command(
                                    command,
                                    pipe_input=pipe_input,
                                    debug=debug,
                                    logger=logger,
                                )

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
                    logger.info("[STDOUT] %s", stdout)
                else:
                    print(stdout)
            return True

    except Exception as e:
        # Catch and log any unexpected runtime exceptions (e.g., permission issues)
        if logger:
            logger.exception("[!] Failed to run command: %s", e)
        else:
            print(f"[!] Failed to run command: {e}")
        return False


def build_adjacency_matrix_fast(df_atoms, df_bonds):
    """
    Build adjacency matrix using optimized indexing for O(n) performance.

    Args:
        df_atoms: DataFrame with atom information
        df_bonds: DataFrame with bond information

    Returns:
        adjacency: numpy array adjacency matrix
        atom_id_to_idx: mapping from atom_id to DataFrame index
    """
    n_atoms = len(df_atoms)
    adjacency = np.zeros((n_atoms, n_atoms), dtype=int)

    # Create atom_id to index mapping for O(1) lookups
    atom_id_to_idx = {atom_id: idx for idx, atom_id in enumerate(df_atoms["atom_id"])}

    # Build adjacency matrix from bonds in O(bonds) time
    for _, bond in df_bonds.iterrows():
        origin_idx = atom_id_to_idx[bond["origin_atom_id"]]
        target_idx = atom_id_to_idx[bond["target_atom_id"]]
        adjacency[origin_idx, target_idx] = 1
        adjacency[target_idx, origin_idx] = 1

    return adjacency, atom_id_to_idx


def build_spatial_grid(coords, max_bond_distance):
    """
    Build a 3D spatial grid for efficient neighbor searching.

    Args:
        coords: numpy array of atom coordinates (n_atoms, 3)
        max_bond_distance: maximum distance to consider for bonds

    Returns:
        grid: dictionary mapping grid coordinates to atom indices
        grid_size: size of each grid cell
    """
    # Calculate grid size based on maximum bond distance
    grid_size = max_bond_distance * 2

    # Find bounding box
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    # Build grid
    grid = {}
    for i, coord in enumerate(coords):
        grid_x = int((coord[0] - min_coords[0]) / grid_size)
        grid_y = int((coord[1] - min_coords[1]) / grid_size)
        grid_z = int((coord[2] - min_coords[2]) / grid_size)
        grid_key = (grid_x, grid_y, grid_z)

        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append(i)

    return grid, grid_size


def get_neighbor_cells(grid_key):
    """Get all neighboring grid cells for a given cell."""
    x, y, z = grid_key
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                neighbors.append((x + dx, y + dy, z + dz))
    return neighbors


def find_bonds_spatial(coords, elements, covalent_radii, bond_tolerance, logger=None):
    """
    Find bonds using spatial partitioning for O(n) average case performance.

    Args:
        coords: numpy array of atom coordinates
        elements: array of element symbols
        covalent_radii: dictionary of covalent radii
        bond_tolerance: bond distance tolerance

    Returns:
        bonds: list of bond dictionaries
        atom_bonds: dictionary tracking bonds per atom
    """
    # Calculate maximum bond distance
    max_radii = max(covalent_radii.values())
    max_bond_distance = 2 * max_radii + bond_tolerance

    # Build spatial grid
    grid, grid_size = build_spatial_grid(coords, max_bond_distance)

    bonds = []
    atom_bonds = {i: [] for i in range(len(coords))}
    bond_id = 1

    # Check each atom against atoms in neighboring cells
    for i in range(len(coords)):
        coord_i = coords[i]
        elem_i = elements[i]

        # Find grid cell for this atom
        min_coords = np.min(coords, axis=0)
        grid_x = int((coord_i[0] - min_coords[0]) / grid_size)
        grid_y = int((coord_i[1] - min_coords[1]) / grid_size)
        grid_z = int((coord_i[2] - min_coords[2]) / grid_size)
        current_cell = (grid_x, grid_y, grid_z)

        # Check atoms in current and neighboring cells
        neighbor_cells = get_neighbor_cells(current_cell)
        checked_atoms = set()

        for cell_key in neighbor_cells:
            if cell_key not in grid:
                continue

            for j in grid[cell_key]:
                if j <= i or j in checked_atoms:  # Avoid duplicates
                    continue
                checked_atoms.add(j)

                elem_j = elements[j]
                coord_j = coords[j]

                # Calculate distance
                dist = np.linalg.norm(coord_i - coord_j)

                # Check if atoms could form a bond
                r_cov_i = covalent_radii.get(elem_i, 0.77)
                r_cov_j = covalent_radii.get(elem_j, 0.77)
                max_bond = r_cov_i + r_cov_j + bond_tolerance

                if 0.4 < dist < max_bond:
                    # Validate bond is chemically possible
                    if is_valid_bond(elem_i, elem_j, atom_bonds, i, j):
                        bonds.append(
                            {
                                "bond_id": bond_id,
                                "origin_atom_id": i + 1,  # 1-based indexing
                                "target_atom_id": j + 1,
                                "bond_type": "1",
                                "status_bit": "",
                            }
                        )
                        # Update bond tracking
                        atom_bonds[i].append(j)
                        atom_bonds[j].append(i)
                        bond_id += 1
                    else:
                        if logger:
                            logger.warning(
                                f"Skipping invalid bond between {elem_i} and {elem_j} (atoms {i + 1} and {j + 1})"
                            )

    return bonds, atom_bonds


def is_valid_bond(elem_i, elem_j, atom_bonds, i, j):
    """
    Check if a bond between two atoms is chemically valid.

    Args:
        elem_i, elem_j: Element symbols
        atom_bonds: Dictionary with lists of bonded atoms
        i, j: Atom indices

    Returns:
        bool: True if bond is chemically valid
    """
    # Prevent H-H bonds (hydrogen can only bond to one other atom)
    if elem_i == "H" and elem_j == "H":
        return False

    # Prevent multiple bonds to hydrogen (hydrogen can only have one bond)
    if elem_i == "H":
        # Check if hydrogen already has any bonds
        if len(atom_bonds[i]) > 0:
            return False

    if elem_j == "H":
        # Check if hydrogen already has any bonds
        if len(atom_bonds[j]) > 0:
            return False

    # Check valence limits for both atoms
    if not check_valence_limits(elem_i, elem_j, atom_bonds, i, j):
        return False

    # Check for common invalid combinations
    invalid_pairs = [
        ("H", "H"),    # H-H bonds
        ("F", "F"),    # F-F bonds (rare and unstable)
        ("CL", "CL"),  # Cl-Cl bonds (rare in organic molecules)
        ("BR", "BR"),  # Br-Br bonds (rare in organic molecules)
        ("I", "I"),    # I-I bonds (rare in organic molecules)
    ]

    for invalid_i, invalid_j in invalid_pairs:
        if (elem_i == invalid_i and elem_j == invalid_j) or (
            elem_i == invalid_j and elem_j == invalid_i
        ):
            return False

    return True


def check_valence_limits(elem_i, elem_j, atom_bonds, i, j):
    """
    Check if adding this bond would exceed valence limits for either atom.
    """
    # Get current valence for both atoms
    valence_i = len(atom_bonds[i])
    valence_j = len(atom_bonds[j])

    # Define maximum valence for each element
    max_valence = {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
        "F": 1,
        "P": 5,
        "S": 6,
        "CL": 1,
        "BR": 1,
        "I": 1,
    }

    # Check if adding this bond would exceed valence
    if elem_i in max_valence and valence_i >= max_valence[elem_i]:
        return False
    if elem_j in max_valence and valence_j >= max_valence[elem_j]:
        return False

    return True

def complete_filename(text, suffix, line=None, begidx=None, endidx=None):
    """
    Generic TAB Autocomplete for filenames in the current directory matching a suffix.

    Parameters:
        text (str): The current input text to match.
        suffix (str): The file suffix or pattern to match (e.g., ".pdb", "solv.ions.gro").
    """
    if not text:
        return [f for f in os.listdir() if f.endswith(suffix)]
    return [f for f in os.listdir() if f.startswith(text) and f.endswith(suffix)]
