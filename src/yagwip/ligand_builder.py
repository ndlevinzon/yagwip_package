"""
ligand_pipeline.py: Ligand parameterization and force field generation pipeline for YAGWIP.
"""

import os
import re
import subprocess
import shutil
from datetime import date
import pandas as pd
import numpy as np
from .base import YagwipBase
from .utils import build_adjacency_matrix_fast, find_bonds_spatial
from .log_utils import auto_monitor


class LigandPipeline(YagwipBase):
    """Ligand parameterization and force field generation pipeline."""

    def __init__(self, logger=None, debug=False):
        """Initialize LigandPipeline."""
        super().__init__(debug=debug, logger=logger)

    @auto_monitor
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

        # Valence rules and atom type assignments
        valence_rules = {
            "C": {"max_valence": 4, "common_types": ["C.3", "C.2", "C.1", "C.ar"]},
            "N": {
                "max_valence": 3,
                "common_types": ["N.3", "N.2", "N.1", "N.ar", "N.am"],
            },
            "O": {"max_valence": 2, "common_types": ["O.3", "O.2", "O.co2"]},
            "S": {"max_valence": 6, "common_types": ["S.3", "S.2", "S.o", "S.o2"]},
            "P": {"max_valence": 5, "common_types": ["P.3"]},
            "F": {"max_valence": 1, "common_types": ["F"]},
            "CL": {"max_valence": 1, "common_types": ["Cl"]},
            "BR": {"max_valence": 1, "common_types": ["Br"]},
            "I": {"max_valence": 1, "common_types": ["I"]},
            "H": {"max_valence": 1, "common_types": ["H"]},
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
            self._log_error(f"[ERROR] No atoms found in {pdb_file}.")
            return None
        df_atoms = pd.DataFrame(atom_records)

        # Bond detection with validation
        coords = df_atoms[["x", "y", "z"]].values
        elements = df_atoms["atom_type"].values
        n_atoms = len(df_atoms)
        # Use spatial partitioning for O(n) bond detection
        bonds, atom_bonds = find_bonds_spatial(
            coords, elements, covalent_radii, bond_tolerance, self.logger
        )
        df_bonds = pd.DataFrame(bonds)

        # Apply valence rules and assign proper atom types
        df_atoms = self._apply_valence_rules(df_atoms, df_bonds, valence_rules)

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
                    f"{int(row['atom_id']):>6d} {row['atom_name']:<8s} {row['x']:>10.4f} {row['y']:>10.4f} "
                    f"{row['z']:>10.4f} {row['atom_type']:<9s} {int(row['subst_id']):<2d} {row['subst_name']:<7s} "
                    f"{row['charge']:>10.4f} {row['status_bit']}\n"
                )
            if len(df_bonds) > 0:
                out_file.write("@<TRIPOS>BOND\n")
                for _, row in mol2["BOND"].iterrows():
                    out_file.write(
                        f"{int(row['bond_id']):>6d} {int(row['origin_atom_id']):>6d} {int(row['target_atom_id']):>6d}"
                        f"    {row['bond_type']} {row['status_bit']}\n"
                    )
        self._log_info(
            f"[SUMMARY] Atoms: {len(df_atoms)}. Bonds: {len(df_bonds)}. MOL2 written to {mol2_file}."
        )
        return mol2_file

    def _apply_valence_rules(self, df_atoms, df_bonds, valence_rules):
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
        adjacency, atom_id_to_idx = build_adjacency_matrix_fast(df_atoms, df_bonds)

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
                atom_type = self._assign_atom_type(
                    element, valence, common_types, df_atoms, i, adjacency
                )
                df_atoms.at[i, "atom_type"] = atom_type
            else:
                # Unknown element, keep original
                pass

        return df_atoms

    def _assign_atom_type(
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
                if self._is_aromatic_carbon(df_atoms, atom_idx, adjacency):
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
                if self._is_amide_nitrogen(df_atoms, atom_idx, adjacency):
                    return "N.am"
                else:
                    return "N.3"  # sp3 nitrogen
            elif valence == 2:
                if self._is_aromatic_nitrogen(df_atoms, atom_idx, adjacency):
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
                if self._is_carbonyl_oxygen(df_atoms, atom_idx, adjacency):
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

    def _is_aromatic_carbon(self, df_atoms, atom_idx, adjacency):
        """Check if carbon is part of an aromatic ring."""
        # Simple heuristic: if carbon has 3 bonds and neighbors are C/N, likely aromatic
        neighbors = np.where(adjacency[atom_idx] == 1)[0]
        if len(neighbors) == 3:
            neighbor_elements = [df_atoms.iloc[n]["atom_type"] for n in neighbors]
            if all(elem in ["C", "N"] for elem in neighbor_elements):
                return True
        return False

    def _is_aromatic_nitrogen(self, df_atoms, atom_idx, adjacency):
        """Check if nitrogen is part of an aromatic ring."""
        # Similar to aromatic carbon
        neighbors = np.where(adjacency[atom_idx] == 1)[0]
        if len(neighbors) == 2:
            neighbor_elements = [df_atoms.iloc[n]["atom_type"] for n in neighbors]
            if all(elem in ["C", "N"] for elem in neighbor_elements):
                return True
        return False

    def _is_amide_nitrogen(self, df_atoms, atom_idx, adjacency):
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

    def _is_carbonyl_oxygen(self, df_atoms, atom_idx, adjacency):
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

    def _get_current_valence(self, atom_bonds, atom_idx):
        """
        Calculate current valence for an atom based on existing bonds.
        """
        return len(atom_bonds[atom_idx])

    def mol2_dataframe_to_orca_charge_input(
        self, df_atoms, output_file, charge=0, multiplicity=1
    ):
        """Generate an ORCA input file from a DataFrame of atomic coordinates."""
        orca_dir = os.path.abspath("orca")
        if not os.path.exists(orca_dir):
            os.makedirs(orca_dir)
        output_file = os.path.abspath(
            os.path.join(orca_dir, os.path.basename(output_file))
        )
        if not {"atom_type", "x", "y", "z"}.issubset(df_atoms.columns):
            raise ValueError(
                "df_atoms must contain 'atom_type', 'x', 'y', 'z' columns."
            )
        with open(output_file, "w") as f:
            f.write(f"!PM3 MINIS MBIS\n")
            f.write(f"* xyz {charge} {multiplicity}\n")
            for _, row in df_atoms.iterrows():
                element = row["atom_type"].split(".")[0]
                f.write(
                    f"{element:2s}  {row['x']:>10.6f}  {row['y']:>10.6f}  {row['z']:>10.6f}\n"
                )
            f.write("*\n")
        self._log_info(f"ORCA input written to: {output_file}")
        return output_file

    def run_orca(self, input_file, output_file=None):
        """Run ORCA quantum chemistry calculation."""
        orca_dir = os.path.abspath("orca")
        if not os.path.exists(orca_dir):
            os.makedirs(orca_dir)
        input_file = os.path.abspath(
            os.path.join(orca_dir, os.path.basename(input_file))
        )
        # Get ToolChecker instance with configuration
        from .config import get_tool_checker
        tool_checker = get_tool_checker()

        orca_path = tool_checker.check_orca_available()  # Check if ORCA is available
        if orca_path is None:
            return False
        openmpi_path = (
            tool_checker.check_openmpi_available()
        )  # Check if OpenMPI is available
        if openmpi_path is None:
            return False
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
                self._log_error(
                    f"[ERROR] ORCA execution failed. See {output_file} for details."
                )
                return False
            self._log_info(
                f"ORCA Charge Calculation Completed. Output: {output_file}\n"
                "Please cite ORCA: Neese, F. Software update: the ORCA program system – DOI: 10.1002/wcms.70019"
            )
            return True
        except Exception as e:
            self._log_error(f"[ERROR] Failed to run ORCA: {e}")
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
        mol2_columns = [
            "atom_id",
            "atom_name",
            "x",
            "y",
            "z",
            "atom_type",
            "subst_id",
            "subst_name",
            "charge",
        ]
        df_atoms = pd.DataFrame(
            [line.split(None, 9) for line in atom_lines], columns=mol2_columns
        )
        df_atoms[["atom_id"]] = df_atoms[["atom_id"]].astype(int)
        df_atoms[["x", "y", "z", "charge"]] = df_atoms[
            ["x", "y", "z", "charge"]
        ].astype(float)

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
        for line in prop_lines[start_idx + 2 :]:
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
        df_charges = pd.DataFrame(
            {
                "index": range(1, len(charge_values) + 1),  # mol2 is 1-based
                "charge": charge_values,
            }
        )

        # Merge the charges into the atom DataFrame using atom_id (mol2) and index (charges)
        df_atoms = df_atoms.merge(
            df_charges,
            left_on="atom_id",
            right_on="index",
            suffixes=("", "_from_property"),
        )

        # Replace the original 'charge' column with the new charge values from the property file
        df_atoms["charge"] = df_atoms["charge_from_property"]
        df_atoms.drop(columns=["index", "charge_from_property"], inplace=True)

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
                for i, line in enumerate(lines[atom_start + 1 :], atom_start + 1):
                    if line.strip().startswith("@<TRIPOS>"):
                        bond_start = i
                        break

                if bond_start is not None:
                    for line in lines[bond_start:]:
                        f.write(line)

        self._log_info(f"Updated charges in {output_path} using {property_path}")
        self._log_info(
            f"[SUMMARY] Updated {len(df_atoms)} atoms with charges from ORCA calculation."
        )

    def run_acpype(self, mol2_file):
        """
        Run ACPYPE to generate topology files from a .mol2 file.
        Args:
            mol2_file (str): Path to the input .mol2 file.
        """
        # Get ToolChecker instance with configuration
        from .config import get_tool_checker
        tool_checker = get_tool_checker()

        acpype_path = tool_checker.check_acpype_available()
        if acpype_path is None:
            self._log_warning("[ERROR] ACPYPE not found. Skipping acpype step.")
            return False
        cmd = f"{acpype_path} -i {mol2_file} -c user"
        self._log_info(f"[RUNNING] {cmd}")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                self._log_error(f"[ERROR] ACPYPE failed: {result.stderr}")
                return False
            self._log_info(f"ACPYPE completed for {mol2_file}")

            # Post-process ACPYPE output files
            self.copy_acpype_output_files(mol2_file)
            return True
        except Exception as e:
            self._log_error(f"[ERROR] Failed to run ACPYPE: {e}")
            return False

    def copy_acpype_output_files(self, mol2_file):
        """
        Copy and rename ACPYPE output files from ligand.acpype subdirectory.
        Args:
            mol2_file (str): Path to the input .mol2 file.
        """
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(mol2_file))[0]
        acpype_dir = f"{base_name}.acpype"

        # Define source and destination files
        source_files = {f"{base_name}_GMX.gro": f"{base_name}.gro", f"{base_name}_GMX.itp": f"{base_name}.itp"}

        if not os.path.exists(acpype_dir):
            self._log_warning(
                f"[ERROR] ACPYPE output directory {acpype_dir} not found."
            )
            return False

        copied_files = []
        for source_file, dest_file in source_files.items():
            source_path = os.path.join(acpype_dir, source_file)
            dest_path = dest_file

            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    self._log_info(f"Copied {source_file} -> {dest_file}")
                    copied_files.append(dest_file)
                except Exception as e:
                    self._log_error(f"[ERROR] Failed to copy {source_file}: {e}")
            else:
                self._log_warning(
                    f"[WARNING] ACPYPE output file {source_file} not found in {acpype_dir}"
                )

        if copied_files:
            self._log_info(
                f"[SUMMARY] Successfully copied {len(copied_files)} files: {', '.join(copied_files)}"
            )
        else:
            self._log_warning("[WARNING] No ACPYPE output files were copied.")

        return len(copied_files) > 0
