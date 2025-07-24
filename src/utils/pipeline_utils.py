# === Standard Library Imports ===
import os
import re
import shutil
from importlib.resources import files

# === Third-Party Imports ===
import numpy as np

# === Local Imports ===
from yagwip.base import LoggingMixin


class Editor(LoggingMixin):
    def __init__(
            self, ligand_itp="ligand.itp", ffnonbonded_itp="./amber14sb.ff/ffnonbonded.itp"
    ):
        self.ligand_itp = ligand_itp
        self.ffnonbonded_itp = ffnonbonded_itp
        self.logger = None

    def _get_ffnonbonded_path(self, itp_file=None):
        """
        Determine the correct ffnonbonded.itp path based on the context.
        For FEP cases, the amber14sb.ff directory is in the same directory as the ITP file.
        For regular cases, use the default path.
        """
        if itp_file is None:
            itp_file = self.ligand_itp

        # Check if this is a FEP case (hybrid.itp in A_to_B or B_to_A directory)
        itp_dir = os.path.dirname(os.path.abspath(itp_file))
        itp_basename = os.path.basename(itp_file)

        # If it's hybrid.itp and we're in a FEP directory structure
        if itp_basename == "hybrid.itp" and ("A_to_B" in itp_dir or "B_to_A" in itp_dir):
            # The amber14sb.ff should be in the same directory as the hybrid.itp
            fep_ff_path = os.path.join(itp_dir, "amber14sb.ff", "ffnonbonded.itp")
            if os.path.exists(fep_ff_path):
                return fep_ff_path
            else:
                self._log(f"[WARNING] FEP forcefield path not found: {fep_ff_path}")

        # Default to the original path
        return self.ffnonbonded_itp

    def append_ligand_atomtypes_to_forcefield(self, itp_file=None, ligand_name=None):
        """
        Append the [ atomtypes ] section from a ligand .itp file to the forcefield ffnonbonded.itp,
        with a comment indicating the ligand name above the block. If itp_file or ligand_name is not provided,
        use self.ligand_itp and infer ligand_name from the file name.
        """
        if itp_file is None:
            itp_file = self.ligand_itp
        if ligand_name is None:
            ligand_name = os.path.splitext(os.path.basename(itp_file))[0]

        if not os.path.isfile(itp_file):
            self._log(f"[!] {itp_file} not found.")
            return

        with open(itp_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        atomtypes_block = []
        inside_atomtypes = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[ atomtypes ]"):
                inside_atomtypes = True
                continue
            if inside_atomtypes and stripped.startswith("["):
                inside_atomtypes = False
            if inside_atomtypes:
                atomtypes_block.append(line)

        if not atomtypes_block:
            self._log(f"No atomtypes section found in {itp_file}. Skipping...")
            return

        if not os.path.isfile(self.ffnonbonded_itp):
            self._log(f"[ERROR] {self.ffnonbonded_itp} not found.")
            return

        # Check if this ligand's atomtypes have already been appended
        with open(self.ffnonbonded_itp, "r", encoding="utf-8") as f:
            ff_content = f.read()
            if f";ligand {ligand_name}" in ff_content:
                self._log(
                    f"Ligand {ligand_name} atomtypes already exist in ffnonbonded.itp. Skipping..."
                )
                return

        # Append the new ligand's atomtypes block to the forcefield file
        with open(self.ffnonbonded_itp, "a", encoding="utf-8") as f:
            f.write(f"\n;ligand {ligand_name}\n")
            f.writelines(atomtypes_block)

        self._log(f"Appended {ligand_name} atomtypes to {self.ffnonbonded_itp}")

    def modify_improper_dihedrals_in_ligand_itp(self):
        if not os.path.isfile(self.ligand_itp):
            self._log(f"[ERROR] {self.ligand_itp} not found.")
            return

        with open(self.ligand_itp, "r", encoding="utf-8") as f:
            lines = f.readlines()

        output_lines = []
        in_dihedrals = False
        modified = False

        for line in lines:
            stripped = line.strip()

            if stripped.lower().startswith("[ dihedrals ]") and "impropers" in stripped:
                in_dihedrals = True
                output_lines.append(line)
                continue

            if in_dihedrals and stripped.startswith("["):
                in_dihedrals = False

            if in_dihedrals and stripped and not stripped.startswith(";"):
                parts = line.split(";")
                comment = f";{parts[1]}" if len(parts) > 1 else ""
                tokens = parts[0].split()

                if len(tokens) >= 8 and tokens[4] == "4":
                    tokens[4] = "2"
                    del tokens[7]
                    new_line = "   " + "   ".join(tokens) + "   " + comment + "\n"
                    output_lines.append(new_line)
                    modified = True
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

        if not modified:
            self._log("No impropers with func=4 found to modify. Skipping...")
            return

        with open(self.ligand_itp, "w", encoding="utf-8") as f:
            f.writelines(output_lines)

        self._log(f"Improper dihedrals converted to func=2 in {self.ligand_itp}.")

    def rename_residue_in_itp_atoms_section(self, old_resname="MOL", new_resname="LIG"):
        with open(self.ligand_itp, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_atoms = in_moleculetype = False
        modified_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("[ atoms ]"):
                in_atoms = True
                modified_lines.append(line)
                continue

            if stripped.startswith("[ moleculetype ]"):
                in_moleculetype = True
                modified_lines.append(line)
                continue

            if in_atoms:
                if stripped == "" or stripped.startswith("["):
                    in_atoms = False
                    modified_lines.append(line)
                    continue
                if not stripped.startswith(";"):
                    parts = re.split(r"(\s+)", line)
                    if len(parts) >= 9 and parts[6].strip() == old_resname:
                        parts[6] = new_resname
                    line = "".join(parts)
                modified_lines.append(line)
                continue

            if in_moleculetype:
                if stripped == "" or stripped.startswith(";"):
                    modified_lines.append(line)
                    continue
                tokens = line.split()
                if len(tokens) >= 2:
                    tokens[0] = new_resname
                    line = f"{tokens[0]:<20}{tokens[1]}\n"
                else:
                    line = f"{new_resname}\n"
                modified_lines.append(line)
                in_moleculetype = False
                continue

            modified_lines.append(line)

        with open(self.ligand_itp, "w", encoding="utf-8") as f:
            f.writelines(modified_lines)

        self._log(
            f"Updated residue names in {self.ligand_itp}: {old_resname} -> {new_resname}"
        )

    def append_ligand_coordinates_to_gro(
            self, protein_gro, ligand_pdb, ligand_itp, combined_gro="complex.gro"
    ):
        """
        Append ligand coordinates to protein GRO file, using atom names from ligand_itp ([ atoms ] section) to ensure consistency with topology.
        Args:
            protein_gro (str): Path to protein .gro file
            ligand_pdb (str): Path to ligand .pdb file
            ligand_itp (str): Path to ligand .itp file (for atom names)
            combined_gro (str): Output .gro file
        """

        # Parse atom names from ligand .itp
        def parse_itp_atom_names(itp_path):
            atom_names = []
            with open(itp_path, "r") as f:
                in_atoms = False
                for line in f:
                    if line.strip().startswith("[ atoms ]"):
                        in_atoms = True
                        continue
                    if in_atoms:
                        if line.strip().startswith("[") or not line.strip():
                            break
                        if not line.strip().startswith(";"):
                            parts = line.split()
                            if len(parts) >= 5:
                                atom_names.append(parts[4])
            return atom_names

        ligand_atom_names = parse_itp_atom_names(ligand_itp)

        # Parse ligand coordinates from PDB
        coords = {}
        with open(ligand_pdb, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    res_id = int(line[23:26].strip())
                    atom_index = int(line[6:11].strip())
                    res_name = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[atom_index] = (res_id, res_name, atom_index, x, y, z)

        if len(coords) != len(ligand_atom_names):
            self._log(
                f"[ERROR] Ligand atom count mismatch: {len(coords)} coords vs {len(ligand_atom_names)} atom names in {ligand_itp}"
            )
            raise ValueError("Ligand atom count mismatch between PDB and ITP")

        # Read protein GRO
        with open(protein_gro, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header, atom_lines, box = lines[:2], lines[2:-1], lines[-1]
        total_atoms = len(atom_lines) + len(coords)

        # Write combined GRO file
        with open(combined_gro, "w", encoding="utf-8") as fout:
            fout.write(header[0])
            fout.write(f"{total_atoms}\n")
            fout.writelines(atom_lines)
            for i, (res_id, res_name, atom_index, x, y, z) in enumerate(coords.values()):
                atom_name = ligand_atom_names[i] if i < len(ligand_atom_names) else "X"
                fout.write(
                    f"{res_id:5d}{res_name:<5}{atom_name:>5}{atom_index:5d}{x / 10:8.3f}{y / 10:8.3f}{z / 10:8.3f}\n"
                )
            fout.write(box)

        self._log(
            f"Wrote combined coordinates to {combined_gro} with ligand atom names from {ligand_itp}"
        )

    def comment_out_topol_line_and_rerun_genions(self, run_genions_func, error_message, max_retries=10):
        """
        If an error like 'ERROR 1 [file topol.top, line 37791]' is encountered during genions,
        comment out the offending line in topol.top and rerun genions, up to max_retries times.
        Args:
            run_genions_func: Callable that runs genions and returns (success: bool, error_message: str)
            error_message: The error message from the last genions run
            max_retries: Maximum number of attempts
        Returns:
            True if genions succeeds, False otherwise
        """
        attempt = 0
        while attempt < max_retries:
            match = re.search(r"\[file topol\\.top, line (\d+)\]", error_message)
            if not match:
                self._log(f"[ERROR] Could not find line number in error message: {error_message}")
                return False
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
                        self._log(f"[#] Commented out line {line_num} in topol.top (Attempt {attempt+1}/{max_retries})")
                else:
                    self._log(f"[ERROR] Line number {line_num} out of range in topol.top")
                    return False
            except Exception as e:
                self._log(f"[ERROR] Failed to modify topol.top: {e}")
                return False
            # Rerun genions
            success, new_error_message = run_genions_func()
            if success:
                self._log(f"[SUCCESS] genions completed successfully after {attempt+1} attempts.")
                return True
            # If error persists, check if it's the same type
            if "[file topol.top, line" in new_error_message:
                error_message = new_error_message
                attempt += 1
                continue
            else:
                self._log(f"[ERROR] genions failed with a different error after {attempt+1} attempts.")
                return False
        self._log(f"[ERROR] Maximum retries ({max_retries}) reached for genions improper dihedral error.")
        return False

    def include_ligand_itp_in_topol(
            self, topol_top="topol.top", ligand_name="LIG", ligand_itp_path=None
    ):
        if ligand_itp_path is None:
            ligand_itp_path = self.ligand_itp
        with open(topol_top, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        inserted_include = False
        inserted_mol = False
        in_molecules_section = False
        molecules_lines = []

        for line in lines:
            stripped = line.strip()

            if (
                    not inserted_include
                    and "#include" in stripped
                    and "forcefield.itp" in stripped
            ):
                new_lines.append(line)
                new_lines.append(f'#include "./{ligand_itp_path}"\n')
                inserted_include = True
                continue

            if "[ molecules ]" in stripped:
                in_molecules_section = True
                molecules_lines.append(line)
                continue

            if in_molecules_section:
                if stripped == "" or stripped.startswith("["):
                    if not inserted_mol:
                        molecules_lines.append(f"{ligand_name}    1\n")
                        inserted_mol = True
                    new_lines.extend(molecules_lines)
                    molecules_lines = []
                    in_molecules_section = False
                    new_lines.append(line)
                else:
                    if ligand_name not in stripped:
                        molecules_lines.append(line)
            else:
                new_lines.append(line)

        if in_molecules_section and not inserted_mol:
            molecules_lines.append(f"{ligand_name}    1\n")
            new_lines.extend(molecules_lines)

        with open(topol_top, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        self._log(f"Included {ligand_itp_path} and {ligand_name} entry in {topol_top}")

    def prepare_fep_directories(self):
        """
        Prepare FEP directories for SLURM FEP workflows.

        This function:
        1. Enters ligand_only/A_to_B and ligand_only/B_to_A directories
        2. Modifies topol.top include paths to be one level deeper
        3. Copies modified topol.top to each lambda directory
        4. Copies and patches MDP files with lambda indices
        5. Copies appropriate .solv.ions.gro files to each lambda directory
        6. Repeats the same procedure for protein_complex directories
        """

        # Define lambda values and their indices
        lambda_values = [
            "0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45", "0.50",
            "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"
        ]

        # FEP MDP templates to copy
        fep_mdp_templates = [
            "em_fep.mdp",
            "nvt_fep.mdp",
            "npt_fep.mdp",
            "production_fep.mdp"
        ]

        # Store original working directory
        original_cwd = os.getcwd()

        try:
            # Process ligand_only directories
            if os.path.isdir("ligand_only"):
                self._log_info("Processing ligand_only FEP directories...")

                # Process A_to_B directory
                a_to_b_dir = os.path.join("ligand_only", "A_to_B")
                if os.path.isdir(a_to_b_dir):
                    self._log_info(f"Processing {a_to_b_dir}...")
                    os.chdir(a_to_b_dir)
                    self._prepare_fep_subdirectory("hybrid_stateA", lambda_values, fep_mdp_templates)
                    os.chdir(original_cwd)

                # Process B_to_A directory
                b_to_a_dir = os.path.join("ligand_only", "B_to_A")
                if os.path.isdir(b_to_a_dir):
                    self._log_info(f"Processing {b_to_a_dir}...")
                    os.chdir(b_to_a_dir)
                    self._prepare_fep_subdirectory("hybrid_stateB", lambda_values, fep_mdp_templates)
                    os.chdir(original_cwd)

            # Process protein_complex directories
            if os.path.isdir("protein_complex"):
                self._log_info("Processing protein_complex FEP directories...")

                # Process A_to_B directory
                a_to_b_dir = os.path.join("protein_complex", "A_to_B")
                if os.path.isdir(a_to_b_dir):
                    self._log_info(f"Processing {a_to_b_dir}...")
                    os.chdir(a_to_b_dir)
                    self._prepare_fep_subdirectory("complex", lambda_values, fep_mdp_templates)
                    os.chdir(original_cwd)

                # Process B_to_A directory
                b_to_a_dir = os.path.join("protein_complex", "B_to_A")
                if os.path.isdir(b_to_a_dir):
                    self._log_info(f"Processing {b_to_a_dir}...")
                    os.chdir(b_to_a_dir)
                    self._prepare_fep_subdirectory("complex", lambda_values, fep_mdp_templates)
                    os.chdir(original_cwd)

            self._log_success("FEP directory preparation completed successfully")

        except Exception as e:
            self._log_error(f"Error during FEP directory preparation: {e}")
            os.chdir(original_cwd)
            raise

    def _prepare_fep_subdirectory(self, gro_base_name, lambda_values, fep_mdp_templates):
        """
        Prepare a single FEP subdirectory (A_to_B or B_to_A).

        Args:
            gro_base_name (str): Base name for .solv.ions.gro file (hybrid_stateA, hybrid_stateB, or complex)
            lambda_values (list): List of lambda value strings
            fep_mdp_templates (list): List of MDP template names
        """
        # Step 1: Modify topol.top include paths to be one level deeper
        if os.path.exists("topol.top"):
            self._log_info("Modifying topol.top include paths...")
            self._modify_topol_paths("topol.top")

        # Step 2: Copy modified topol.top to each lambda directory
        for i, lambda_val in enumerate(lambda_values):
            lambda_dir = f"lambda_{lambda_val}"
            if os.path.isdir(lambda_dir):
                self._log_info(f"Processing {lambda_dir} (index {i})...")

                # Copy modified topol.top
                if os.path.exists("topol.top"):
                    shutil.copy2("topol.top", os.path.join(lambda_dir, "topol.top"))

                # Step 3: Copy and patch MDP files
                self._copy_and_patch_mdp_files(lambda_dir, fep_mdp_templates, i)

                # Step 4: Copy appropriate .solv.ions.gro file
                gro_file = f"{gro_base_name}.solv.ions.gro"
                if os.path.exists(gro_file):
                    shutil.copy2(gro_file, os.path.join(lambda_dir, gro_file))
                    self._log_info(f"Copied {gro_file} to {lambda_dir}")
                else:
                    self._log_warning(f"{gro_file} not found for {lambda_dir}")

    def _modify_topol_paths(self, topol_file):
        """
        Modify include paths in topol.top to be one level deeper.

        Args:
            topol_file (str): Path to topol.top file
        """
        with open(topol_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace include paths to be one level deeper
        # From: #include "./amber14sb.ff/..."
        # To:   #include "../amber14sb.ff/..."
        modified_content = content.replace(
            '#include "./amber14sb.ff/',
            '#include "../amber14sb.ff/'
        )
        modified_content = modified_content.replace(
            '#include "./hybrid.itp"',
            '#include "../hybrid.itp"'
        )

        with open(topol_file, "w", encoding="utf-8") as f:
            f.write(modified_content)

        self._log_info(f"Modified include paths in {topol_file}")

    def _copy_and_patch_mdp_files(self, lambda_dir, mdp_templates, lambda_index):
        """
        Copy and patch MDP files for a specific lambda directory.

        Args:
            lambda_dir (str): Lambda directory path
            mdp_templates (list): List of MDP template names
            lambda_index (int): Lambda index for substitution
        """

        template_dir = files("templates")

        for mdp_name in mdp_templates:
            src_path = template_dir / mdp_name

            if not src_path.is_file():
                self._log_warning(f"MDP template {mdp_name} not found in templates")
                continue

            try:
                # Read template content
                with open(str(src_path), "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace XXX with lambda index
                content = content.replace("XXX", str(lambda_index))

                # Write to lambda directory
                dest_path = os.path.join(lambda_dir, mdp_name)
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self._log_info(f"Wrote {mdp_name} to {lambda_dir} with lambda index {lambda_index}")

            except Exception as e:
                self._log_error(f"Failed to copy MDP file {mdp_name} to {lambda_dir}: {e}")


class LigandUtils(LoggingMixin):
    def build_adjacency_matrix_fast(self, df_atoms, df_bonds):
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
        atom_id_to_idx = {
            atom_id: idx for idx, atom_id in enumerate(df_atoms["atom_id"])
        }

        # Build adjacency matrix from bonds in O(bonds) time
        for _, bond in df_bonds.iterrows():
            origin_idx = atom_id_to_idx[bond["origin_atom_id"]]
            target_idx = atom_id_to_idx[bond["target_atom_id"]]
            adjacency[origin_idx, target_idx] = 1
            adjacency[target_idx, origin_idx] = 1

        return adjacency, atom_id_to_idx

    def build_spatial_grid(self, coords, max_bond_distance):
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

    def get_neighbor_cells(self, grid_key):
        """Get all neighboring grid cells for a given cell."""
        x, y, z = grid_key
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbors.append((x + dx, y + dy, z + dz))
        return neighbors

    def find_bonds_spatial(
            self, coords, elements, covalent_radii, bond_tolerance, connect_records=None, logger=None
    ):
        """
        Find bonds using spatial partitioning for O(n) average case performance.
        If connect_records is provided, use it as the authoritative source for all bonds for those atoms.
        """
        max_radii = max(covalent_radii.values())
        max_bond_distance = 2 * max_radii + bond_tolerance
        grid, grid_size = self.build_spatial_grid(coords, max_bond_distance)
        n_atoms = len(coords)
        bonds = []
        atom_bonds = {i: [] for i in range(n_atoms)}
        bond_id = 1
        used_pairs = set()

        # --- 1. Use connect_records if provided ---
        if connect_records:
            # connect_records: dict of {atom_index (1-based): [bonded_atom_indices (1-based)]}
            for i1_str, bonded_list in connect_records.items():
                i1 = int(i1_str) - 1  # convert to 0-based
                for i2 in bonded_list:
                    i2 = int(i2) - 1
                    pair = tuple(sorted((i1, i2)))
                    if pair in used_pairs:
                        continue
                    used_pairs.add(pair)
                    bonds.append({
                        "bond_id": bond_id,
                        "origin_atom_id": i1 + 1,
                        "target_atom_id": i2 + 1,
                        "bond_type": "1",
                        "status_bit": "",
                    })
                    atom_bonds[i1].append(i2)
                    atom_bonds[i2].append(i1)
                    bond_id += 1
            # Now, for atoms not in connect_records, assign bonds by distance (e.g., hydrogens)
            for i in range(n_atoms):
                if (i + 1) in connect_records:
                    continue  # already handled
                coord_i = coords[i]
                elem_i = elements[i]
                min_coords = np.min(coords, axis=0)
                grid_x = int((coord_i[0] - min_coords[0]) / grid_size)
                grid_y = int((coord_i[1] - min_coords[1]) / grid_size)
                grid_z = int((coord_i[2] - min_coords[2]) / grid_size)
                current_cell = (grid_x, grid_y, grid_z)
                neighbor_cells = self.get_neighbor_cells(current_cell)
                checked_atoms = set()
                for cell_key in neighbor_cells:
                    if cell_key not in grid:
                        continue
                    for j in grid[cell_key]:
                        if j <= i or j in checked_atoms:
                            continue
                        checked_atoms.add(j)
                        elem_j = elements[j]
                        coord_j = coords[j]
                        dist = np.linalg.norm(coord_i - coord_j)
                        r_cov_i = covalent_radii.get(elem_i, 0.77)
                        r_cov_j = covalent_radii.get(elem_j, 0.77)
                        max_bond = r_cov_i + r_cov_j + bond_tolerance
                        pair = tuple(sorted((i, j)))
                        if pair in used_pairs:
                            continue
                        if 0.4 < dist < max_bond:
                            if self.is_valid_bond(elem_i, elem_j, atom_bonds, i, j):
                                bonds.append({
                                    "bond_id": bond_id,
                                    "origin_atom_id": i + 1,
                                    "target_atom_id": j + 1,
                                    "bond_type": "1",
                                    "status_bit": "",
                                })
                                atom_bonds[i].append(j)
                                atom_bonds[j].append(i)
                                used_pairs.add(pair)
                                bond_id += 1
                            else:
                                if logger:
                                    logger.warning(
                                        f"Skipping invalid bond between {elem_i} and {elem_j} (atoms {i + 1} and {j + 1})"
                                    )
            return bonds, atom_bonds

        # --- 2. Default: distance-based assignment for all atoms ---
        for i in range(n_atoms):
            coord_i = coords[i]
            elem_i = elements[i]
            min_coords = np.min(coords, axis=0)
            grid_x = int((coord_i[0] - min_coords[0]) / grid_size)
            grid_y = int((coord_i[1] - min_coords[1]) / grid_size)
            grid_z = int((coord_i[2] - min_coords[2]) / grid_size)
            current_cell = (grid_x, grid_y, grid_z)
            neighbor_cells = self.get_neighbor_cells(current_cell)
            checked_atoms = set()
            for cell_key in neighbor_cells:
                if cell_key not in grid:
                    continue
                for j in grid[cell_key]:
                    if j <= i or j in checked_atoms:
                        continue
                    checked_atoms.add(j)
                    elem_j = elements[j]
                    coord_j = coords[j]
                    dist = np.linalg.norm(coord_i - coord_j)
                    r_cov_i = covalent_radii.get(elem_i, 0.77)
                    r_cov_j = covalent_radii.get(elem_j, 0.77)
                    max_bond = r_cov_i + r_cov_j + bond_tolerance
                    pair = tuple(sorted((i, j)))
                    if pair in used_pairs:
                        continue
                    if 0.4 < dist < max_bond:
                        if self.is_valid_bond(elem_i, elem_j, atom_bonds, i, j):
                            bonds.append({
                                "bond_id": bond_id,
                                "origin_atom_id": i + 1,
                                "target_atom_id": j + 1,
                                "bond_type": "1",
                                "status_bit": "",
                            })
                            atom_bonds[i].append(j)
                            atom_bonds[j].append(i)
                            used_pairs.add(pair)
                            bond_id += 1
                        else:
                            if logger:
                                logger.warning(
                                    f"Skipping invalid bond between {elem_i} and {elem_j} (atoms {i + 1} and {j + 1})"
                                )
        return bonds, atom_bonds

    def is_valid_bond(self, elem_i, elem_j, atom_bonds, i, j):
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
        if not self.check_valence_limits(elem_i, elem_j, atom_bonds, i, j):
            return False

        # Check for common invalid combinations
        invalid_pairs = [
            ("H", "H"),  # H-H bonds
            ("F", "F"),  # F-F bonds (rare and unstable)
            ("CL", "CL"),  # Cl-Cl bonds (rare in organic molecules)
            ("BR", "BR"),  # Br-Br bonds (rare in organic molecules)
            ("I", "I"),  # I-I bonds (rare in organic molecules)
        ]

        for invalid_i, invalid_j in invalid_pairs:
            if (elem_i == invalid_i and elem_j == invalid_j) or (
                    elem_i == invalid_j and elem_j == invalid_i
            ):
                return False

        return True

    def check_valence_limits(self, elem_i, elem_j, atom_bonds, i, j):
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

    def apply_valence_rules(self, df_atoms, df_bonds, valence_rules):
        """
        Apply valence rules to assign proper atom types and bond orders.

        Args:
            df_atoms: DataFrame with atom information
            df_bonds: DataFrame with bond information
            valence_rules: Dictionary with valence rules for each element

        Returns:
            DataFrame with updated atom types
        """
        # Use optimized adjacency matrix construction
        adjacency, atom_id_to_idx = LigandUtils().build_adjacency_matrix_fast(
            df_atoms, df_bonds
        )

        # Calculate valence for each atom
        valence_counts = np.sum(adjacency, axis=1)

        # Assign atom types based on valence rules
        for i, atom in df_atoms.iterrows():
            element = atom["atom_type"]
            valence = valence_counts[i]

            if element in valence_rules:
                rules = valence_rules[element]
                max_valence = rules["max_valence"]
                common_types = rules["common_types"]

                # Check if valence exceeds maximum
                if valence > max_valence:
                    self._log_warning(
                        f"[WARNING] Atom {atom['atom_id']} ({element}) has valence {valence} > {max_valence}"
                    )

                # Assign atom type based on element and valence
                atom_type = self.assign_atom_type(
                    element, valence, common_types, df_atoms, i, adjacency
                )
                df_atoms.at[i, "atom_type"] = atom_type
            else:
                # Unknown element, keep original
                pass

        return df_atoms

    def assign_atom_type(
            self, element, valence, common_types, df_atoms, atom_idx, adjacency
    ):
        """
        Assign specific atom type based on element, valence, and local environment.
        """
        if element == "C":
            if valence == 4:
                return "C.3"  # sp3 carbon
            elif valence == 3:
                # Check if it's aromatic or has double bonds
                if self.is_aromatic_carbon(df_atoms, atom_idx, adjacency):
                    return "C.ar"
                else:
                    return "C.2"  # sp2 carbon
            elif valence == 2:
                return "C.1"  # sp carbon
            else:
                return "C.3"  # default to sp3

        elif element == "N":
            if valence == 3:
                # Check for amide nitrogen
                if self.is_amide_nitrogen(df_atoms, atom_idx, adjacency):
                    return "N.am"
                else:
                    return "N.3"  # sp3 nitrogen
            elif valence == 2:
                if self.is_aromatic_nitrogen(df_atoms, atom_idx, adjacency):
                    return "N.ar"
                else:
                    return "N.2"  # sp2 nitrogen
            elif valence == 1:
                return "N.1"  # sp nitrogen
            else:
                return "N.3"  # default

        elif element == "O":
            if valence == 2:
                # Check if it's carbonyl oxygen
                if self.is_carbonyl_oxygen(df_atoms, atom_idx, adjacency):
                    return "O.2"  # carbonyl oxygen
                else:
                    return "O.3"  # sp3 oxygen
            elif valence == 1:
                return "O.3"  # sp3 oxygen
            else:
                return "O.3"  # default

        elif element == "S":
            if valence == 2:
                return "S.2"  # sp2 sulfur
            elif valence == 1:
                return "S.3"  # sp3 sulfur
            else:
                return "S.3"  # default

        elif element == "P":
            return "P.3"  # sp3 phosphorus

        elif element in ["F", "CL", "BR", "I"]:
            return element if element != "CL" else "Cl"

        elif element == "H":
            return "H"

        else:
            # Unknown element, return as is
            return element

    def is_aromatic_carbon(self, df_atoms, atom_idx, adjacency):
        """Check if carbon is part of an aromatic ring."""
        # Simple heuristic: if carbon has 3 bonds and neighbors are C/N, likely aromatic
        neighbors = np.where(adjacency[atom_idx] == 1)[0]
        if len(neighbors) == 3:
            neighbor_elements = [df_atoms.iloc[n]["atom_type"] for n in neighbors]
            if all(elem in ["C", "N"] for elem in neighbor_elements):
                return True
        return False

    def is_aromatic_nitrogen(self, df_atoms, atom_idx, adjacency):
        """Check if nitrogen is part of an aromatic ring."""
        # Similar to aromatic carbon
        neighbors = np.where(adjacency[atom_idx] == 1)[0]
        if len(neighbors) == 2:
            neighbor_elements = [df_atoms.iloc[n]["atom_type"] for n in neighbors]
            if all(elem in ["C", "N"] for elem in neighbor_elements):
                return True
        return False

    def is_amide_nitrogen(self, df_atoms, atom_idx, adjacency):
        """Check if nitrogen is part of an amide group."""
        neighbors = np.where(adjacency[atom_idx] == 1)[0]
        for neighbor_idx in neighbors:
            neighbor = df_atoms.iloc[neighbor_idx]
            if neighbor["atom_type"] == "C":
                # Check if this carbon has a double bond to oxygen
                carbon_neighbors = np.where(adjacency[neighbor_idx] == 1)[0]
                for carbon_neighbor_idx in carbon_neighbors:
                    carbon_neighbor = df_atoms.iloc[carbon_neighbor_idx]
                    if carbon_neighbor["atom_type"] == "O":
                        return True
        return False

    def is_carbonyl_oxygen(self, df_atoms, atom_idx, adjacency):
        """Check if oxygen is part of a carbonyl group (C=O)."""
        neighbors = np.where(adjacency[atom_idx] == 1)[0]
        for neighbor_idx in neighbors:
            neighbor = df_atoms.iloc[neighbor_idx]
            if neighbor["atom_type"] == ["C", "S"]:
                # Check if this carbon has only 3 bonds total (indicating double bond)
                carbon_valence = np.sum(adjacency[neighbor_idx])
                if carbon_valence == 3:  # C=O bond
                    return True
        return False

    def get_current_valence(self, atom_bonds, atom_idx):
        """
        Calculate current valence for an atom based on existing bonds.
        """
        return len(atom_bonds[atom_idx])
