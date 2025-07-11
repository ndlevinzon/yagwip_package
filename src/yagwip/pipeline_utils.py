import os
import re
from .base import LoggingMixin
import numpy as np


class Editor(LoggingMixin):
    def __init__(
            self, ligand_itp="ligand.itp", ffnonbonded_itp="./amber14sb.ff/ffnonbonded.itp"
    ):
        self.ligand_itp = ligand_itp
        self.ffnonbonded_itp = ffnonbonded_itp
        self.logger = None

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
                self._log(f"Ligand {ligand_name} atomtypes already exist in ffnonbonded.itp. Skipping...")
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
            self, protein_gro, ligand_pdb, combined_gro="complex.gro"
    ):
        coords = []
        with open(ligand_pdb, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    res_id = int(line[23:26].strip())
                    atom_name = line[13:16].strip()
                    res_name = line[17:20].strip()
                    atom_index = int(line[6:11].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((res_id, res_name, atom_name, atom_index, x, y, z))

        with open(protein_gro, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header, atom_lines, box = lines[:2], lines[2:-1], lines[-1]
        total_atoms = len(atom_lines) + len(coords)

        with open(combined_gro, "w", encoding="utf-8") as fout:
            fout.write(header[0])
            fout.write(f"{total_atoms}\n")
            fout.writelines(atom_lines)
            for res_id, res_name, atom_name, atom_index, x, y, z in coords:
                fout.write(
                    f"{res_id:5d}{res_name:<5}{atom_name:>5}{atom_index:5d}{x / 10:8.3f}{y / 10:8.3f}{z / 10:8.3f}\n"
                )
            fout.write(box)

        self._log(f"Wrote combined coordinates to {combined_gro}")

    def append_hybrid_ligand_coordinates_to_gro(
            self, protein_gro, ligand_pdb, hybrid_itp, combined_gro="complex.gro"
    ):
        """
        Append hybrid ligand coordinates to protein GRO file, ensuring atom indices match the topology.
        This function reads the hybrid ITP file to get the correct atom indices and order.
        """
        # Read hybrid ITP to get atom indices and order
        hybrid_atoms = []
        with open(hybrid_itp, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_atoms_section = False
        for line in lines:
            if line.strip() == "[ atoms ]":
                in_atoms_section = True
                continue
            elif in_atoms_section and line.strip().startswith("["):
                break
            elif in_atoms_section and line.strip() and not line.strip().startswith(";"):
                parts = line.split()
                if len(parts) >= 4:
                    atom_index = int(parts[0])
                    atom_name = parts[4]
                    hybrid_atoms.append((atom_index, atom_name))

        # Read ligand PDB coordinates
        coords = {}
        with open(ligand_pdb, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    res_id = int(line[23:26].strip())
                    atom_name = line[13:16].strip()
                    res_name = line[17:20].strip()
                    atom_index = int(line[6:11].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[atom_index] = (res_id, res_name, atom_name, x, y, z)

        # Create ordered coordinates based on hybrid ITP atom order
        ordered_coords = []
        for topo_index, topo_name in hybrid_atoms:
            # Find matching coordinate by atom name (since PDB indices are sequential)
            for pdb_index, (res_id, res_name, atom_name, x, y, z) in coords.items():
                if atom_name.strip() == topo_name.strip():
                    ordered_coords.append((res_id, res_name, atom_name, topo_index, x, y, z))
                    break

        # Read protein GRO file
        with open(protein_gro, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header, atom_lines, box = lines[:2], lines[2:-1], lines[-1]
        total_atoms = len(atom_lines) + len(ordered_coords)

        # Write combined GRO file
        with open(combined_gro, "w", encoding="utf-8") as fout:
            fout.write(header[0])
            fout.write(f"{total_atoms}\n")
            fout.writelines(atom_lines)
            for res_id, res_name, atom_name, atom_index, x, y, z in ordered_coords:
                fout.write(
                    f"{res_id:5d}{res_name:<5}{atom_name:>5}{atom_index:5d}{x / 10:8.3f}{y / 10:8.3f}{z / 10:8.3f}\n"
                )
            fout.write(box)

        self._log(f"Wrote combined coordinates to {combined_gro} with topology-matched indices")
        self._log(f"Protein atoms: {len(atom_lines)}, Ligand atoms: {len(ordered_coords)}, Total: {total_atoms}")

        # Debug: Check if all hybrid atoms were matched
        if len(ordered_coords) != len(hybrid_atoms):
            self._log(
                f"[WARNING] Atom count mismatch: {len(ordered_coords)} coordinates vs {len(hybrid_atoms)} topology atoms")
            missing_atoms = []
            for topo_index, topo_name in hybrid_atoms:
                found = False
                for _, _, atom_name, _, _, _ in ordered_coords:
                    if atom_name.strip() == topo_name.strip():
                        found = True
                        break
                if not found:
                    missing_atoms.append(f"{topo_index}:{topo_name}")
            if missing_atoms:
                self._log(f"[WARNING] Missing atoms in PDB: {missing_atoms}")

    def include_ligand_itp_in_topol(self, topol_top="topol.top", ligand_name="LIG", ligand_itp_path=None):
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

    def fix_lambda_topology_paths(self, topol_file, lambda_value):
        """
        Fix paths in topology files for lambda subdirectories.
        - Replace #include "./amber14sb.ff/" with #include "../amber14sb.ff/"
        - Replace #include "./ligand" with #include "./hybrid_lambda_X.itp"
        - Replace #include "./posre.itp" with #include "../posre.itp"

        Parameters:
            topol_file (str): Path to the topology file to fix
            lambda_value (str): Lambda value (e.g., "0.00", "0.05", etc.)
        """
        if not os.path.isfile(topol_file):
            self._log(f"[ERROR] {topol_file} not found.")
            return

        with open(topol_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified_lines = []
        modified = False

        for line in lines:
            # Fix amber14sb.ff path
            if '#include "./amber14sb.ff/' in line:
                line = line.replace('./amber14sb.ff/', '../amber14sb.ff/')
                modified = True
                self._log(f"Fixed amber14sb.ff path in {topol_file}")

            # Fix ligand include to use hybrid ITP
            if '#include "./ligand' in line:
                line = f'#include "./hybrid_lambda_{lambda_value}.itp"\n'
                modified = True
                self._log(f"Updated ligand include to hybrid_lambda_{lambda_value}.itp in {topol_file}")

            # Fix amber14sb.ff path
            if '#include "posre.itp' in line:
                line = line.replace('posre.itp', '../posre.itp')
                modified = True
                self._log(f"Fixed amber14sb.ff path in {topol_file}")

            modified_lines.append(line)

        if modified:
            with open(topol_file, "w", encoding="utf-8") as f:
                f.writelines(modified_lines)
            self._log(f"Updated paths in {topol_file} for lambda {lambda_value}")
        else:
            self._log(f"No path corrections needed in {topol_file}")

    def insert_itp_into_top_files(self, itp_path_list, root_dir="."):
        """
        Rewrite all topol.top files to include only the provided list of ITP paths,
        removing any existing non-user-specified includes. Used for the SOURCE command.

        Parameters:
            itp_path_list (list): List of ITP file paths (strings) to include in the topology files.
            root_dir (str): Root directory to search for topol.top files (default: current directory).
        """
        top_files = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file == "topol.top":
                    top_files.append(os.path.join(root, file))

        for top_file in top_files:
            with open(top_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            inserted = False

            for line in lines:
                # Remove all #include lines that aren't forcefield.itp
                if line.strip().startswith("#include") and "forcefield" not in line:
                    continue
                new_lines.append(line)

            # Find insertion point (after forcefield include)
            insert_idx = 0
            for i, line in enumerate(new_lines):
                if "#include" in line and "forcefield" in line:
                    insert_idx = i + 1
                    break

            # Inject all custom includes
            include_lines = [f'#include "{path}"\n' for path in itp_path_list]
            new_lines[insert_idx:insert_idx] = include_lines

            # Write updated file
            with open(top_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            self._log(f"Injected {len(itp_path_list)} custom includes into {top_file}")

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
        atom_id_to_idx = {atom_id: idx for idx, atom_id in enumerate(df_atoms["atom_id"])}

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

    def find_bonds_spatial(self, coords, elements, covalent_radii, bond_tolerance, logger=None):
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
        grid, grid_size = self.build_spatial_grid(coords, max_bond_distance)

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
            neighbor_cells = self.get_neighbor_cells(current_cell)
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
                        if self.is_valid_bond(elem_i, elem_j, atom_bonds, i, j):
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



