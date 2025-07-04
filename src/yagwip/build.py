"""
build.py: GROMACS and ligand building utilities for YAGWIP.
"""
# === Standard Library Imports ===
import os
import re
import subprocess
import shutil
from datetime import date
from importlib.resources import files

# === Third-Party Imports ===
import pandas as pd
import numpy as np

# === Local Imports ===
from .utils import run_gromacs_command, LoggingMixin, ToolChecker

# Constants for GROMACS command inputs
PIPE_INPUTS = {"pdb2gmx": "1\n",
               "genion_prot": "13\n",
               "genion_complex": "15\n"
               }


class Builder(LoggingMixin):
    """Handles GROMACS system building steps."""

    def __init__(self, gmx_path, debug=False, logger=None):
        """Initialize Builder."""
        self.gmx_path = gmx_path
        self.debug = debug
        self.logger = logger

    def _resolve_basename(self, basename):
        """Resolve the basename for file operations."""
        if not basename and not self.debug:
            msg = "[ERROR] No PDB loaded. Use `loadPDB <filename.pdb>` first."
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)
            return None
        return basename if basename else "PLACEHOLDER"

    def run_pdb2gmx(self, basename, custom_command=None):
        """Run pdb2gmx to generate topology and coordinates."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        command = custom_command or (
            f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        )
        if self.debug:
            print(f"[DEBUG] Command: {command}")
            return
        self._log(f"Running pdb2gmx for {base}.pdb...")
        run_gromacs_command(
            command,
            pipe_input=PIPE_INPUTS["pdb2gmx"],
            debug=self.debug,
            logger=self.logger,
        )

    def run_solvate(self, basename, arg="", custom_command=None):
        """Run solvate to add solvent to the system."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        default_box = " -c -d 1.0 -bt cubic"
        default_water = "spc216.gro"
        parts = arg.strip().split()
        box_options = parts[0] if len(parts) > 0 else default_box
        water_model = parts[1] if len(parts) > 1 else default_water
        default_cmds = [
            f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro{box_options}",
            f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs {water_model} -o {base}.solv.gro -p topol.top",
        ]
        if self.debug:
            for cmd in default_cmds:
                print(f"[DEBUG] Command: {cmd}")
            return
        if custom_command:
            self._log("[CUSTOM] Using custom solvate command")
            run_gromacs_command(custom_command, debug=self.debug, logger=self.logger)
        else:
            for cmd in default_cmds:
                run_gromacs_command(cmd, debug=self.debug, logger=self.logger)

    def run_genions(self, basename, custom_command=None):
        """Run genion to add ions to the system."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        default_ions = files("yagwip.templates").joinpath("ions.mdp")
        input_gro = f"{base}.solv.gro"
        output_gro = f"{base}.solv.ions.gro"
        tpr_out = "ions.tpr"
        ion_options = "-pname NA -nname CL -conc 0.150 -neutral"
        grompp_opts = ""
        ion_pipe_input = (
            PIPE_INPUTS["genion_prot"]
            if base.endswith("protein")
            else PIPE_INPUTS["genion_complex"]
        )
        default_cmds = [
            f"{self.gmx_path} grompp -f {default_ions} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts} -maxwarn 50",
            f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}",
        ]
        if self.debug:
            for cmd in default_cmds:
                print(f"[DEBUG] Command: {cmd}")
            return
        self._log(f"Running genion for {base}...")
        if custom_command:
            self._log("[CUSTOM] Using custom genion command")
            run_gromacs_command(custom_command, debug=self.debug, logger=self.logger)
        else:
            for cmd in default_cmds:
                run_gromacs_command(
                    cmd, pipe_input=ion_pipe_input, debug=self.debug, logger=self.logger
                )


class Modeller(LoggingMixin):
    """Protein structure modeller for missing residues."""

    def __init__(self, pdb, logger=None, debug=False, output_file="protein_test.pdb"):
        """Initialize Modeller."""
        self.logger = logger
        self.debug = debug
        self.pdb = pdb
        self.output_file = output_file

    def find_missing_residues(self):
        """Identifies missing internal residues by checking for gaps in residue numbering."""
        self._log(f"Checking for missing residues in {self.pdb}...")
        residue_map = {}  # {chain_id: sorted list of residue IDs}
        with open(self.pdb, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain_id = line[21].strip() or "A"
                    try:
                        res_id = int(line[22:26].strip())
                    except ValueError:
                        continue
                    if chain_id not in residue_map:
                        residue_map[chain_id] = set()
                    residue_map[chain_id].add(res_id)
        gaps = []
        for chain_id, residues in residue_map.items():
            sorted_residues = sorted(residues)
            for i in range(len(sorted_residues) - 1):
                current = sorted_residues[i]
                next_expected = current + 1
                if sorted_residues[i + 1] != next_expected:
                    gaps.append((chain_id, current, sorted_residues[i + 1]))
        self._log(
            f"[WARNING] Found missing residue ranges: {gaps}" if gaps else "No gaps found."
        )
        return gaps


class LigandPipeline(LoggingMixin):
    """Ligand parameterization and force field generation pipeline."""

    def __init__(self, logger=None, debug=False):
        """Initialize LigandPipeline."""
        self.logger = logger
        self.debug = debug

    def convert_pdb_to_mol2(self, pdb_file, mol2_file=None):
        """Converts a ligand PDB file to a MOL2 file using a custom parser and writer."""
        # Covalent radii in Ångstroms for common elements (extend as needed)
        covalent_radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "F": 0.57,
            "P": 1.07,
            "S": 1.05,
            "CL": 1.02,
            "BR": 1.20,
            "I": 1.39,
        }
        bond_tolerance = 0.45  # Ångstroms
        if mol2_file is None:
            mol2_file = pdb_file.replace(".pdb", ".mol2")
        # Efficiently parse ATOM/HETATM lines from PDB
        atom_records = []
        with open(pdb_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    atom_id = int(line[6:11])
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    element = line[76:78].strip().upper() if len(line) >= 78 else ""
                    if not element:
                        # Fallback: use first letter of atom_name
                        element = atom_name[0].upper()
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26])
                    chain_id = line[21].strip() or "A"
                    atom_records.append(
                        {
                            "atom_id": atom_id,
                            "atom_name": atom_name,
                            "x": x,
                            "y": y,
                            "z": z,
                            "atom_type": element,
                            "subst_id": 1,
                            "subst_name": res_name,
                            "charge": 0.0,
                            "status_bit": "",
                        }
                    )
        if not atom_records:
            self._log(f"[ERROR] No atoms found in {pdb_file}.")
            return None
        df_atoms = pd.DataFrame(atom_records)
        # Bond detection
        coords = df_atoms[["x", "y", "z"]].values
        elements = df_atoms["atom_type"].values
        n_atoms = len(df_atoms)
        bonds = []
        bond_id = 1
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                elem_i = elements[i]
                elem_j = elements[j]
                r_cov_i = covalent_radii.get(
                    elem_i, 0.77
                )  # Default to C radius if unknown
                r_cov_j = covalent_radii.get(elem_j, 0.77)
                max_bond = r_cov_i + r_cov_j + bond_tolerance
                dist = np.linalg.norm(coords[i] - coords[j])
                if 0.4 < dist < max_bond:
                    bonds.append(
                        {
                            "bond_id": bond_id,
                            "origin_atom_id": int(df_atoms.iloc[i]["atom_id"]),
                            "target_atom_id": int(df_atoms.iloc[j]["atom_id"]),
                            "bond_type": "1",  # single bond for now
                            "status_bit": "",
                        }
                    )
                    bond_id += 1
        df_bonds = pd.DataFrame(bonds)
        # Build minimal MOL2 dict
        mol2 = {}
        mol2["MOLECULE"] = pd.DataFrame(
            [
                {
                    "mol_name": os.path.splitext(os.path.basename(pdb_file))[0],
                    "num_atoms": len(df_atoms),
                    "num_bonds": len(df_bonds),
                    "num_subst": 1,
                    "num_feat": 0,
                    "num_sets": 0,
                    "mol_type": "SMALL",
                    "charge_type": "NO_CHARGES",
                }
            ]
        )
        mol2["ATOM"] = df_atoms
        mol2["BOND"] = df_bonds
        # Write MOL2 file
        with open(mol2_file, "w", encoding="utf-8") as out_file:
            today = date.today().strftime("%Y-%m-%d")
            out_file.write(f"### Crafted by Yagwip LigandPipeline {today}\n")
            out_file.write("### Charged by ORCA 6.1 \n\n")
            out_file.write("@<TRIPOS>MOLECULE\n")
            m = mol2["MOLECULE"].iloc[0]
            out_file.write(f"{m['mol_name']}\n")
            out_file.write(
                f" {m['num_atoms']} {m['num_bonds']} {m['num_subst']} {m['num_feat']} {m['num_sets']}\n"
            )
            out_file.write(f"{m['mol_type']}\n")
            out_file.write(f"{m['charge_type']}\n\n")
            out_file.write("@<TRIPOS>ATOM\n")
            for _, row in mol2["ATOM"].iterrows():
                out_file.write(
                    f"{int(row['atom_id']):>6d} {row['atom_name']:<8s} {row['x']:>10.4f} {row['y']:>10.4f} {row['z']:>10.4f} {row['atom_type']:<9s} {int(row['subst_id']):<2d} {row['subst_name']:<7s} {row['charge']:>10.4f} {row['status_bit']}\n"
                )
            if len(df_bonds) > 0:
                out_file.write("@<TRIPOS>BOND\n")
                for _, row in mol2["BOND"].iterrows():
                    out_file.write(
                        f"{int(row['bond_id']):>6d} {int(row['origin_atom_id']):>6d} {int(row['target_atom_id']):>6d}    {row['bond_type']} {row['status_bit']}\n"
                    )
        self._log(
            f"[SUMMARY] Atoms: {len(df_atoms)}. Bonds: {len(df_bonds)}. MOL2 written to {mol2_file}."
        )
        return mol2_file

    def mol2_dataframe_to_orca_charge_input(self, df_atoms, output_file, charge=0, multiplicity=1):
        """Generate an ORCA input file from a DataFrame of atomic coordinates."""
        # Use absolute path for output file
        output_file = os.path.abspath(output_file)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not {"atom_type", "x", "y", "z"}.issubset(df_atoms.columns):
            raise ValueError(
                "df_atoms must contain 'atom_type', 'x', 'y', 'z' columns."
            )
        with open(output_file, "w") as f:
            f.write(f"!PM3 MINIS MBIS\n")
            f.write(f"* xyz {charge} {multiplicity}\n")
            for _, row in df_atoms.iterrows():
                element = row["atom_type"]
                f.write(
                    f"{element:2s}  {row['x']:>10.6f}  {row['y']:>10.6f}  {row['z']:>10.6f}\n"
                )
            f.write("*\n")
        self._log(f"ORCA input written to: {output_file}")
        return output_file

    def run_orca(self, orca_path, input_file, output_file=None):
        """Run ORCA quantum chemistry calculation."""
        input_file = os.path.abspath(input_file)
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + ".out"
        else:
            output_file = os.path.abspath(
                os.path.join(orca_dir, os.path.basename(output_file))
            )
        try:
            result = subprocess.run(
                [orca_path, input_file], capture_output=True, text=True, check=False
            )
            with open(output_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n[STDERR]\n" + result.stderr)
            if result.returncode != 0:
                self._log(f"[ERROR] ORCA execution failed. See {output_file} for details.")
                return False
            self._log(
                f"ORCA Charge Calculation Completed. Output: {output_file}\n"
                "Please cite ORCA: Neese, F. Software update: the ORCA program system – \n"
                "Version 6.0 Wiley Interdiscip. Rev.: Comput. Mol. Sci., 2025, 15, 2, e70019 (DOI: 10.1002/wcms.70019)"
            )
            return True
        except Exception as e:
            self._log(f"[ERROR] Failed to run ORCA: {e}")
            return False

    def apply_orca_charges_to_mol2(self, mol2_path, property_path, output_path=None):
        """
        Update the charges in a MOL2 file using the AtomCharges from an ORCA property file.
        Args:
            mol2_path (str): Path to the .mol2 file.
            property_path (str): Path to the ORCA ligand.property.txt file.
            output_path (str, optional): Path to write the updated .mol2 file. If None, overwrite input.
        """
        # Load mol2 file and extract atom block
        with open(mol2_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        atom_section = False
        atom_lines = []
        for line in lines:
            if line.strip().startswith("@<TRIPOS>ATOM"):
                atom_section = True
                continue
            if line.strip().startswith("@<TRIPOS>"):
                atom_section = False
            if atom_section:
                atom_lines.append(line.strip())

        # Parse ATOM lines into a DataFrame
        mol2_columns = ["atom_id", "atom_name", "x", "y", "z", "atom_type", "subst_id", "subst_name", "charge"]
        df_atoms = pd.DataFrame([line.split(None, 9) for line in atom_lines], columns=mol2_columns)
        df_atoms[["atom_id"]] = df_atoms[["atom_id"]].astype(int)
        df_atoms[["x", "y", "z", "charge"]] = df_atoms[["x", "y", "z", "charge"]].astype(float)

        # Parse charges from property file
        with open(property_path, "r", encoding="utf-8") as file:
            prop_lines = file.readlines()

        # Find the AtomicCharges section
        start_idx = None
        for i, line in enumerate(prop_lines):
            if re.search(r"&AtomicCharges\b", line, re.IGNORECASE):
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("Could not find the '&AtomicCharges' block.")

        # The block starts two lines after the header (skip the extra "0" line)
        charge_values = []
        pattern = re.compile(r"^\s*(\d+)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
        for line in prop_lines[start_idx + 2:]:
            stripped = line.strip()
            if not stripped:
                continue
            match = pattern.match(line)
            if not match:
                break  # end of charge block
            idx, charge_str = match.groups()
            try:
                charge_values.append(float(charge_str))
            except ValueError:
                continue

        # Create the charges DataFrame
        df_charges = pd.DataFrame({
            'index': range(1, len(charge_values) + 1),  # mol2 is 1-based
            'charge': charge_values
        })

        # Merge the charges into the atom DataFrame using atom_id (mol2) and index (charges)
        df_atoms = df_atoms.merge(df_charges, left_on='atom_id', right_on='index', suffixes=('', '_from_property'))

        # Replace the original 'charge' column with the new charge values from the property file
        df_atoms['charge'] = df_atoms['charge_from_property']
        df_atoms.drop(columns=['index', 'charge_from_property'], inplace=True)

        # Write updated mol2 file
        if output_path is None:
            output_path = mol2_path

        with open(output_path, "w", encoding="utf-8") as f:
            # Write everything up to ATOM section
            atom_start = None
            for i, line in enumerate(lines):
                if line.strip() == "@<TRIPOS>ATOM":
                    atom_start = i
                    break
                f.write(line)

            if atom_start is not None:
                # Write ATOM section header
                f.write(lines[atom_start])

                # Write updated atom lines
                for _, row in df_atoms.iterrows():
                    f.write(
                        f"{int(row['atom_id']):>6d} {row['atom_name']:<8s} {row['x']:>10.4f} {row['y']:>10.4f} {row['z']:>10.4f} {row['atom_type']:<9s} {int(row['subst_id']):<2d} {row['subst_name']:<7s} {row['charge']:>10.4f}\n"
                    )

                # Write the rest of the mol2 file (from BOND section onward)
                bond_start = None
                for i, line in enumerate(lines[atom_start + 1:], atom_start + 1):
                    if line.strip().startswith("@<TRIPOS>"):
                        bond_start = i
                        break

                if bond_start is not None:
                    for line in lines[bond_start:]:
                        f.write(line)

        self._log(f"[#] Updated charges in {output_path} using {property_path}")
        self._log(f"[SUMMARY] Updated {len(df_atoms)} atoms with charges from ORCA calculation.")

    def run_parmchk2(self, mol2_file, frcmod_file=None):
        """
        Run AmberTools parmchk2 to generate a .frcmod file from a .mol2 file.
        Args:
            mol2_file (str): Path to the input .mol2 file.
            frcmod_file (str, optional): Path to the output .frcmod file. Defaults to 'ligand.frcmod' in the same directory.
        """
        amber_path = ToolChecker.check_amber_available()
        if amber_path is None:
            self._log("[ERROR] AmberTools (parmchk2) not found. Skipping parmchk2 step.")
            return False
        if frcmod_file is None:
            frcmod_file = os.path.splitext(mol2_file)[0] + ".frcmod"
        cmd = f"{amber_path} -i {mol2_file} -f mol2 -o {frcmod_file}"
        self._log(f"[RUNNING] {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                self._log(f"[ERROR] parmchk2 failed: {result.stderr}")
                return False
            self._log(f"[#] parmchk2 completed. Output: {frcmod_file}")
            return True
        except Exception as e:
            self._log(f"[ERROR] Failed to run parmchk2: {e}")
            return False

    def run_acpype(self, mol2_file):
        """
        Run ACPYPE to generate topology files from a .mol2 file.
        Args:
            mol2_file (str): Path to the input .mol2 file.
        """
        acpype_path = ToolChecker.check_acpype_available()
        if acpype_path is None:
            self._log("[ERROR] ACPYPE not found. Skipping acpype step.")
            return False
        cmd = f"{acpype_path} -i {mol2_file}"
        self._log(f"[RUNNING] {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                self._log(f"[ERROR] acpype failed: {result.stderr}")
                return False
            self._log(f"[#] acpype completed for {mol2_file}")
            return True
        except Exception as e:
            self._log(f"[ERROR] Failed to run acpype: {e}")
            return False
