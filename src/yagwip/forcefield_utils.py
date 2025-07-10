import os
import re
from .base import LoggingMixin


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



