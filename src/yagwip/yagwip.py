"""
yagwip.py: (Y)et (A)nother (G)ROMACS (W)rapper (I)n (P)ython

Copyright (c) 2025 the Authors.
Authors: Nathan Levinzon, Olivier Mailhot

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# === Standard Library Imports ===
import os
import re
import io
from io import StringIO
import cmd
import sys
import string
import subprocess
import random
import argparse
import shutil
import contextlib
import importlib.metadata
import multiprocessing as mp
from pathlib import Path
from importlib.resources import files

# === Third-Party Imports ===
import pandas as pd

# === Local Imports ===
from utils.gromacs_runner import Builder, Sim
from yagwip.ligand_builder import LigandPipeline
from yagwip.base import YagwipBase
from yagwip.config import validate_gromacs_installation
from utils.slurm_utils import SlurmWriter
from utils.pipeline_utils import Editor
from utils.log_utils import setup_logger
from utils.batch_processor import ParallelBatchProcessor

# === Metadata ===
__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YagwipShell(cmd.Cmd, YagwipBase):
    """
    Interactive shell for YAGWIP: Yet Another GROMACS Wrapper In Python.
    Provides a command-line interface for molecular simulation workflows.
    """

    # Intro message and prompt for the interactive CLI
    intro = f"Welcome to YAGWIP v{__version__}. Type help to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        """Initialize the YAGWIP shell with GROMACS path."""
        # Initialize cmd.Cmd first (no parameters)
        cmd.Cmd.__init__(self)
        # Initialize YagwipBase with our parameters
        YagwipBase.__init__(self, gmx_path=gmx_path, debug=False)

        self.current_pdb_path = None  # Full path to the loaded PDB file
        self.ligand_pdb_path = None  # Full path to the ligand PDB file, if any
        self.basename = None  # Base PDB filename (without extension)
        self.print_banner()  # Prints intro banner to command line
        self.user_itp_paths = []  # Stores user input paths for do_source
        self.editor = (
            Editor()
        )  # Initialize the file Editor class from pipeline_utils.py
        self.ligand_pipeline = LigandPipeline(logger=self.logger, debug=self.debug)
        # Initialize the Sim class from sim.py
        self.sim = Sim(gmx_path=self.gmx_path, debug=self.debug, logger=self.logger)

        # Initialize the Builder Pipeline class from gromacs_runner.py
        self.builder = Builder(
            gmx_path=self.gmx_path, debug=self.debug, logger=self.logger
        )
        # Validate GROMACS installation
        try:
            validate_gromacs_installation(gmx_path)
        except RuntimeError as e:
            self._log_error(f"GROMACS Validation Error: {e}")
            self._log_error(
                "YAGWIP cannot start without GROMACS. Please install GROMACS and try again."
            )
            sys.exit(1)
        # Dictionary of custom command overrides set by the user
        self.custom_cmds = {k: "" for k in ("pdb2gmx", "solvate", "genions")}
        self.ligand_counter = 0  # For FEP-style ligand naming
        self.current_ligand_name = None

    def _require_pdb(self):
        """Check if a PDB file is loaded."""
        if not self.current_pdb_path and not self.debug:
            self._log_error("No PDB loaded.")
            return False
        return True

    def default(self, line):
        """Throws error when command is not recognized."""
        self._log_error(f"Unknown command: {line}")

    def do_debug(self, arg):
        """
        Debug Mode: Enhanced logging with detailed resource statistics and command information.
        Commands are still executed, but with verbose output including system resources.

        Usage: Toggle with 'debug', 'debug on', or 'debug off'
        """
        arg = arg.lower().strip()
        if arg == "on":
            self.debug = True
        elif arg == "off":
            self.debug = False
        else:
            self.debug = not self.debug
        # Update logger and simulation mode

        self.logger = setup_logger(debug_mode=self.debug)

        if self.debug:
            self._log_info("Debug mode is now ON")
            self._log_info("Enhanced debug mode will show:")
            self._log_info("  - Detailed command information")
            self._log_info("  - System resource statistics")
            self._log_info("  - Runtime monitoring data")
            self._log_info("  - All commands will still be executed")
        else:
            self._log_info("Debug mode is now OFF")

    def print_banner(self):
        """Prints YAGWIP Banner Logo on Start."""
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(os.path.dirname(module_dir), "assets")
            banner_path = os.path.join(assets_dir, "yagwip_banner.txt")
            with open(str(banner_path), "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            self._log_error(f"Could not load banner: {e}")

    def do_show(self, arg):
        """Show current custom or default commands."""
        for k in ["pdb2gmx", "solvate", "genions"]:
            cmd_str = self.custom_cmds.get(k)
            self._log_info(f"{k}: {cmd_str if cmd_str else '[DEFAULT]'}")

    def do_runtime(self, arg):
        """Show runtime statistics and performance metrics."""
        if hasattr(self, "runtime_monitor"):
            summary = self.runtime_monitor.get_summary()
            if summary:
                self._log_info("=== Runtime Statistics ===")
                self._log_info(f"Total Operations: {summary['total_operations']}")
                self._log_info(f"Successful: {summary['successful_operations']}")
                self._log_info(f"Failed: {summary['failed_operations']}")
                self._log_info(f"Success Rate: {summary['success_rate']:.1%}")
                self._log_info(
                    f"Total Duration: {summary['total_duration_seconds']:.2f}s"
                )
                self._log_info(
                    f"Average Duration: {summary['average_duration_seconds']:.2f}s"
                )
            else:
                self._log_info("No runtime data available yet.")
        else:
            self._log_info("Runtime monitoring not available.")

    def do_set(self, arg):
        """
        Edit the default command string for pdb2gmx, solvate, or genions.
        Usage:
            set pdb2gmx
            set solvate
            set genions
        The user is shown the current command and can modify it inline.
        Press ENTER to accept the modified command.
        Type 'quit' to cancel.
        """
        valid_keys = ["pdb2gmx", "solvate", "genions"]
        cmd_key = arg.strip().lower()
        if cmd_key not in valid_keys:
            self._log_error(f"Usage: set <{'|'.join(valid_keys)}>")
            return
        # Get the default command string
        base = self.basename if self.basename else "PLACEHOLDER"
        if cmd_key == "pdb2gmx":
            default = (
                f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
            )
        elif cmd_key == "solvate":
            default = (
                f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro -c -d 1.0 -bt cubic && "
                f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs spc216.gro -o {base}.solv.gro -p topol.top"
            )
        elif cmd_key == "genions":
            ions_mdp = "ions.mdp"  # assuming it's copied to current dir already
            default = (
                f"{self.gmx_path} grompp -f {ions_mdp} -c {base}.solv.gro -r {base}.solv.gro -p topol.top -o ions.tpr && "
                f"{self.gmx_path} genion -s ions.tpr -o {base}.solv.ions.gro -p topol.top -pname NA -nname CL -conc 0.150 -neutral"
            )
        # Show current command and prompt for new input
        current = self.custom_cmds.get(cmd_key) or default
        self._log_info(f"Current command for {cmd_key}:\n{current}")
        self._log_info(
            "Type new command or press ENTER to keep current. Type 'quit' to cancel."
        )
        new_cmd = input("New command: ").strip()
        if new_cmd.lower() == "quit":
            self._log_info("Edit canceled.")
            return
        if not new_cmd:
            self.custom_cmds[cmd_key] = current
            self._log_info("Keeping existing command.")
            return
        self.custom_cmds[cmd_key] = new_cmd
        self._log_success(f"Updated command for {cmd_key}.")

    def _complete_filename(self, text, suffix, line=None, begidx=None, endidx=None):
        """
        Generic TAB Autocomplete for filenames in the current directory matching a suffix.

        Parameters:
            text (str): The current input text to match.
            suffix (str): The file suffix or pattern to match (e.g., ".pdb", "solv.ions.gro").
        """
        if not text:
            return [f for f in os.listdir() if f.endswith(suffix)]
        return [f for f in os.listdir() if f.startswith(text) and f.endswith(suffix)]

    def complete_loadpdb(self, text, line=None, begidx=None, endidx=None):
        """Tab completion for .pdb files."""
        return self._complete_filename(text, ".pdb", line, begidx, endidx)

    def do_loadpdb(self, arg):
        """
        Usage: "loadpdb X.pdb [--ligand_builder] [--c CHARGE] [--m MULTIPLICITY] (Requires ORCA)."
                --ligand_builder: Run the ligand building pipeline if ligand.itp is missing.
                --c: Set the total charge for QM input (default 0)
                --m: Set the multiplicity for QM input (default 1)
        """
        args = self._parse_loadpdb_args(arg)
        try:
            lines = self._read_pdb_file(args.pdb_file)
        except FileNotFoundError:
            return
        hetatm_lines, atom_lines = self._split_pdb_lines(lines)

        if hetatm_lines and not atom_lines:
            self._handle_ligand_only(hetatm_lines, args)
            return

        if hetatm_lines:
            self._handle_protein_ligand(lines, hetatm_lines, args)
            return

        self._handle_protein_only(lines)

    def _parse_loadpdb_args(self, arg):
        parser = argparse.ArgumentParser(description="Load PDB file")
        parser.add_argument("pdb_file", help="PDB file to load")
        parser.add_argument(
            "--ligand_builder", action="store_true", help="Use ligand builder"
        )
        parser.add_argument(
            "--c", type=int, default=0, help="Total charge for QM input"
        )
        parser.add_argument(
            "--m", type=int, default=1, help="Multiplicity for QM input"
        )
        return parser.parse_args(arg.split())

    def _read_pdb_file(self, pdb_file):
        full_path = os.path.abspath(pdb_file)
        if not os.path.isfile(full_path):
            self._log_error(f"'{pdb_file}' not found.")
            raise FileNotFoundError
        self.current_pdb_path = full_path
        self.basename = os.path.splitext(os.path.basename(full_path))[0]
        self._log_success(f"PDB file loaded: {full_path}")
        with open(full_path, "r") as f:
            return f.readlines()

    def _split_pdb_lines(self, lines):
        hetatm_lines = [line for line in lines if line.startswith("HETATM")]
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        return hetatm_lines, atom_lines

    def _handle_ligand_only(self, hetatm_lines, args):
        ligand_name = self._assign_ligand_name()
        ligand_file = f"{ligand_name}.pdb"
        self.ligand_pdb_path = os.path.abspath(ligand_file)
        self._write_ligand_pdb(ligand_file, hetatm_lines)
        self._warn_if_no_hydrogens(hetatm_lines)
        itp_file = f"{ligand_name}.itp"
        if os.path.isfile(itp_file):
            self._process_ligand_itp(itp_file, ligand_name)
        elif args.ligand_builder:
            self._run_ligand_builder(ligand_file, ligand_name, args.c, args.m)
            if os.path.isfile(itp_file):
                self._process_ligand_itp(itp_file, ligand_name)
        else:
            self._log_info(f"{itp_file} not found and --ligand_builder not specified.")

    def _handle_protein_ligand(self, lines, hetatm_lines, args):
        ligand_name = self._assign_ligand_name()
        protein_file, ligand_file, connect_records = self._extract_ligand_and_protein_with_connect(lines, ligand_name)
        self.ligand_pdb_path = os.path.abspath(ligand_file)
        self._warn_if_no_hydrogens(hetatm_lines)
        itp_file = f"{ligand_name}.itp"
        if os.path.isfile(itp_file):
            self._process_ligand_itp(itp_file, ligand_name)
        elif args.ligand_builder:
            # Pass CONNECT records to ligand_pipeline
            self._run_ligand_builder(ligand_file, ligand_name, args.c, args.m, connect_records=connect_records)
            if os.path.isfile(itp_file):
                self._process_ligand_itp(itp_file, ligand_name)
        else:
            self._log_info(f"{itp_file} not found and --ligand_builder not specified.")

    def _warn_if_missing_residues(self, protein_pdb):
        """Identifies missing internal residues by checking for gaps in residue numbering."""
        self._log_info(f"Checking for missing residues in {protein_pdb}")
        residue_map = {}  # {chain_id: sorted list of residue IDs}
        with open(protein_pdb, "r", encoding="utf-8") as f:
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
        if gaps:
            self._log_warning(f"Found missing residue ranges: {gaps}")
        else:
            self._log_info("No gaps found.")
        return gaps

    def _handle_protein_only(self, lines):
        self.ligand_pdb_path = None
        with open("protein.pdb", "w", encoding="utf-8") as prot_out:
            for line in lines:
                if line[17:20] in ("HSP", "HSD"):
                    line = line[:17] + "HIS" + line[20:]
                prot_out.write(line)
        self._log_info(
            "No HETATM entries found. Wrote corrected PDB to protein.pdb and using it as apo protein."
        )
        self._warn_if_missing_residues(protein_pdb="protein.pdb")

    def _assign_ligand_name(self):
        ligand_name = chr(ord("A") + self.ligand_counter)
        ligand_name = f"ligand{ligand_name}"
        self.current_ligand_name = ligand_name
        self.ligand_counter += 1
        return ligand_name

    def _write_ligand_pdb(self, ligand_file, hetatm_lines):
        with open(ligand_file, "w", encoding="utf-8") as lig_out:
            for line in hetatm_lines:
                lig_out.write(line[:17] + "LIG" + line[20:])
        self._log_info(
            f"Ligand-only PDB detected. Assigned name: {ligand_file}. Wrote ligand to {ligand_file}"
        )

    def _warn_if_no_hydrogens(self, hetatm_lines):
        if not any(
            line[76:78].strip() == "H" or line[12:16].strip().startswith("H")
            for line in hetatm_lines
        ):
            self._log_warning(
                "Ligand appears to lack hydrogen atoms. Consider checking hydrogens and valences."
            )

    def _run_ligand_builder(self, ligand_file, ligand_name, charge, multiplicity, connect_records=None):
        amber_ff_source = str(files("templates").joinpath("amber14sb.ff/"))
        amber_ff_dest = os.path.abspath("amber14sb.ff")
        if not os.path.exists(amber_ff_dest):
            os.makedirs(amber_ff_dest)
            self._log_info(f"Created directory: {amber_ff_dest}")
            try:
                for item in Path(amber_ff_source).iterdir():
                    if item.is_file():
                        content = item.read_text(encoding="utf-8")
                        dest_file = os.path.join(amber_ff_dest, item.name)
                        with open(dest_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        self._log_debug(f"Copied {item.name}")
                self._log_success("Copied all amber14sb.ff files.")
            except Exception as e:
                self._log_error(f"Failed to copy amber14sb.ff files: {e}")
        else:
            self._log_info(f"amber14sb.ff already exists, not overwriting.")

        # Pass connect_records to convert_pdb_to_mol2
        mol2_file = self.ligand_pipeline.convert_pdb_to_mol2(ligand_file, connect_records=connect_records)
        if not mol2_file or not os.path.isfile(mol2_file):
            self._log_error(
                f"MOL2 generation failed or file not found: {mol2_file}. Aborting ligand pipeline..."
            )
            return

        with open(mol2_file, encoding="utf-8") as f:
            lines = f.readlines()
        atom_start = atom_end = None
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>ATOM":
                atom_start = i + 1
            elif line.strip().startswith("@<TRIPOS>BOND") and atom_start is not None:
                atom_end = i
                break
        if atom_start is None:
            self._log_error("Could not find ATOM section in MOL2 file.")
            return
        if atom_end is None:
            atom_end = len(lines)
        atom_lines = lines[atom_start:atom_end]


        df_atoms = pd.read_csv(
            io.StringIO("".join(atom_lines)),
            sep=r"\s+",
            header=None,
            names=[
                "atom_id",
                "atom_name",
                "x",
                "y",
                "z",
                "atom_type",
                "subst_id",
                "subst_name",
                "charge",
                "status_bit",
            ],
        )
        orca_geom_input = mol2_file.replace(".mol2", ".inp")
        self.ligand_pipeline.mol2_dataframe_to_orca_charge_input(
            df_atoms,
            orca_geom_input,
            charge=charge,
            multiplicity=multiplicity,
        )
        self.ligand_pipeline.run_orca(orca_geom_input)

        # Check if ORCA property file exists before applying charges
        property_file = f"orca/{ligand_name}.property.txt"
        if not os.path.exists(property_file):
            self._log_error(
                f"ORCA property file not found: {property_file}. ORCA calculation may have failed."
            )
            return

        self.ligand_pipeline.apply_orca_charges_to_mol2(mol2_file, property_file)
        self.ligand_pipeline.run_acpype(mol2_file)
        self.ligand_pipeline.copy_acpype_output_files(mol2_file)

    def _process_ligand_itp(self, itp_file, ligand_name):
        self.editor.append_ligand_atomtypes_to_forcefield(itp_file, ligand_name)
        self.editor.ligand_itp = itp_file
        self.editor.modify_improper_dihedrals_in_ligand_itp()
        self.editor.rename_residue_in_itp_atoms_section()

    def _extract_ligand_and_protein_with_connect(self, lines, ligand_name):
        protein_file = "protein.pdb"
        ligand_file = f"{ligand_name}.pdb"
        ligand_indices = []
        connect_records = {}
        with open(protein_file, "w", encoding="utf-8") as prot_out, open(
            ligand_file, "w", encoding="utf-8"
        ) as lig_out:
            for line in lines:
                if line.startswith("HETATM"):
                    lig_out.write(line[:17] + "LIG" + line[20:])
                    try:
                        idx = int(line[6:11])
                        ligand_indices.append(idx)
                    except Exception:
                        pass
                else:
                    if line[17:20] in ("HSP", "HSD"):
                        line = line[:17] + "HIS" + line[20:]
                    prot_out.write(line)
        # Now extract CONNECT records for ligand atoms
        for line in lines:
            if line.startswith("CONECT"):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        idx = int(parts[1])
                        if idx in ligand_indices:
                            bonded = [int(x) for x in parts[2:] if x.isdigit() and int(x) in ligand_indices]
                            if bonded:
                                connect_records[idx] = bonded
                    except Exception:
                        pass
        self._log_info(f"Detected ligand. Split into: {protein_file}, {ligand_file}, with {len(connect_records)} ligand CONNECT records.")
        return protein_file, ligand_file, connect_records

    def do_fep_prep(self, arg):
        """
        Run the FEP preparation workflow using fep_prep.py CLI.
        This will:
        1) Find MCS and write atom_map.txt
        2) Align ligandB.mol2 to ligandA.mol2
        3) Align ligandB.pdb to ligandA.pdb
        4) Organize all files into A/B_complex/water directories
        Output: atom_map.txt, ligandB_aligned.mol2, ligandB_aligned.pdb, and subdirectories.
        """
        cwd = os.getcwd()
        required_files = [
            "ligandA.mol2",
            "ligandB.mol2",
            "ligandA.pdb",
            "ligandA.itp",
            "ligandB.pdb",
            "ligandB.itp",
            "protein.pdb",
        ]
        missing = [f for f in required_files if not os.path.isfile(os.path.join(cwd, f))]
        if missing:
            self._log_error(f"Missing required files: {', '.join(missing)}")
            return
        fep_prep_path = os.path.join(os.path.dirname(__file__), "fep_prep.py")
        python_exe = sys.executable
        cmd = [
            python_exe,
            fep_prep_path,
            "--ligA_mol2", "ligandA.mol2",
            "--ligB_mol2", "ligandB.mol2",
            "--ligA_pdb", "ligandA.pdb",
            "--ligA_itp", "ligandA.itp",
            "--ligB_pdb", "ligandB.pdb",
            "--ligB_itp", "ligandB.itp",
        ]
        self._log_info("FEP prep workflow:")
        self._log_info("  1. Find MCS and write atom_map.txt")
        self._log_info("  2. Align ligandB.mol2 to ligandA.mol2")
        self._log_info("  3. Align ligandB.pdb to ligandA.pdb")
        self._log_info("  4. Organize all files into subdirectories")
        self._log_info(f"Running FEP prep: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._log_success("FEP preparation complete.")
            self._log_info("Output files: atom_map.txt, ligandB_aligned.mol2, ligandB_aligned.pdb, and subdirectories.")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            self._log_error(f"FEP prep failed: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

    def do_pdb2gmx(self, arg):
        """
        Run pdb2gmx. Handles protein-only, protein-ligand, and FEP (A/B) cases.
        """
        if not os.path.isfile("protein.pdb"):
            self._log_info("protein.pdb not found. Switching to ligand-only workflow.")
            # Call _pdb2gmx_ligand with dummy args (lambda_dirs=None, output_gro=None) for now
            self._pdb2gmx_ligand(lambda_dirs=None, output_gro=None)
            return
        amber_ff_source = str(files("templates").joinpath("amber14sb.ff/"))
        amber_ff_dest = os.path.abspath("amber14sb.ff")
        if not os.path.exists(amber_ff_dest):
            os.makedirs(amber_ff_dest)
            self._log_info(f"Created directory: {amber_ff_dest}")
            try:
                for item in Path(amber_ff_source).iterdir():
                    if item.is_file():
                        content = item.read_text(encoding="utf-8")
                        dest_file = os.path.join(amber_ff_dest, item.name)
                        with open(dest_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        self._log_debug(f"Copied {item.name}")
                self._log_success("Copied all amber14sb.ff files.")
            except Exception as e:
                self._log_error(f"Failed to copy amber14sb.ff files: {e}")
        else:
            self._log_info(f"amber14sb.ff already exists, not overwriting.")

        # First, run pdb2gmx on the protein component
        protein_pdb = "protein"
        output_gro = f"{protein_pdb}.gro"
        self.builder.run_pdb2gmx(protein_pdb, custom_command=self.custom_cmds.get("pdb2gmx"))
        if not os.path.isfile(output_gro):
            self._log_error(f"Expected {output_gro} was not created by pdb2gmx.")
            return

        # FEP: Check for A/B complex/water directories
        fep_dirs = ["A_complex", "A_water", "B_complex", "B_water"]
        fep_present = all(os.path.isdir(d) for d in fep_dirs)
        if fep_present:
            self._pdb2gmx_fep(fep_dirs)
        else:
            # Check for ligand presence
            ligand_present = os.path.isfile("ligandA.pdb") or os.path.isfile("ligand.pdb")
            if ligand_present:
                self._pdb2gmx_protein_ligand(output_gro)

    def _pdb2gmx_fep(self, fep_dirs):
        """
        Run pdb2gmx for FEP workflow (A/B complex/water directories).
        """
        self._log_info("Detected FEP directories: {}".format(", ".join(fep_dirs)))
        # TODO: Implement FEP-specific pdb2gmx logic here
        self._log_info("[FEP] pdb2gmx workflow not yet implemented.")

    def _pdb2gmx_protein_ligand(self, protein_gro):
        """Handle the protein-ligand and protein-only workflows for pdb2gmx."""
        ligand_pdb_file = "ligandA.pdb"
        ligand_itp_file = "ligandA.itp"

        # Protein + ligand case
        if self.ligand_pdb_path and os.path.exists(ligand_pdb_file) and os.path.getsize(ligand_pdb_file) > 0:
            if not os.path.exists(ligand_itp_file):
                self._log_error(f"Ligand ITP file not found: {ligand_itp_file}")
                return
            self._log_info("Processing protein-ligand system...")
            # Add ligand coordinates to protein gro and update topology
            self.editor.append_ligand_coordinates_to_gro(protein_gro, ligand_pdb_file, ligand_itp_file, "complex.gro")
            self.editor.include_ligand_itp_in_topol("topol.top", "LIG", ligand_itp_path=ligand_itp_file)

    def _pdb2gmx_ligand(self, lambda_dirs, output_gro):
        """
        Handle the lambda directory workflow for pdb2gmx. Checks for ligandX.pdb in the current directory.
        """
        ligand_pdb_files = [f"ligand{c}.pdb" for c in string.ascii_uppercase]
        found = False
        base = None
        for fname in ligand_pdb_files:
            if os.path.isfile(fname):
                found = True
                base = fname
                break
        if not found:
            self._log_error(f"No ligand_*.pdb file found in current directory. Expected one of: {', '.join(ligand_pdb_files)}")
            return
        # Find the first ligandX.acpype directory and copy the topology
        for c in string.ascii_uppercase:
            acpype_dir = f"ligand{c}.acpype"
            if os.path.isdir(acpype_dir):
                gmx_top = os.path.join(acpype_dir, f"ligand{c}_GMX.top")
                if os.path.isfile(gmx_top):
                    # Read the topology file
                    with open(gmx_top, 'r') as f:
                        content = f.read()

                    # Remove the [ defaults ] block
                    content = re.sub(r'\[ defaults \]\s*\n; nbfunc\s+comb-rule\s+gen-pairs\s+fudgeLJ fudgeQQ\s*\n\d+\s+\d+\s+\w+\s+[\d\.]+\s+[\d\.]+\s*\n',
                                     '', content)

                    # Remove the entire POSRES_LIG block
                    content = re.sub(
                        r'; Ligand position restraints\s*\n#ifdef POSRES_LIG\s*\n#include "posre_[^"]*\.itp"\s*\n#endif\s*\n',
                        '', content)

                    # Remove all "_GMX" strings
                    content = content.replace("_GMX", "")

                    # Replace ligandX with LIG in the [ molecules ] section
                    content = re.sub(r'ligand[A-Z]\s+\d+', 'LIG              1', content)

                    # Replace the ligand GMX itp include line with forcefield include
                    content = re.sub(r'#include "ligand[A-Z]\.itp"\s*\n',
                                     '#include "./amber14sb.ff/forcefield.itp"\n', content)

                    # Write the modified content to topol.top
                    with open("topol.top", 'w') as f:
                        f.write(content)

                    self._log_success(f"Modified and copied {gmx_top} to topol.top")
                    self.editor.include_ligand_itp_in_topol("topol.top", "LIG", ligand_itp_path=None)
                else:
                    self._log_error("No ligand.acpype directory with ligand_GMX.top found.")
                    return

    def do_solvate(self, arg):
        """
        Run solvate with optional custom command override. Handles three cases:
        1) Protein-only: solvates protein.gro
        2) Protein + single ligand: solvates complex.gro
        3) Ligand-only: solvates ligandX.gro in current directory
        4) Lambda subdirectories: solvates hybrid_complex_XX.gro in each lambda directory
        Usage: "solvate"
        Other Options: use "set solvate" to override defaults
        """
        # Determine which system to solvate
        if self.ligand_pdb_path and os.path.isfile("complex.gro"):
            base = "complex"
        elif not os.path.isfile("protein.gro"):
            # Ligand-only: look for ligandX.gro
            ligand_gro_files = [f"ligand{c}.gro" for c in string.ascii_uppercase]
            found = None
            for fname in ligand_gro_files:
                if os.path.isfile(fname):
                    found = fname[:-4]  # strip .gro
                    break
            if found:
                base = found
            else:
                self._log_error("No protein.gro or ligand_*.gro found for solvation.")
                return
        else:
            base = "protein"
        if not self._require_pdb():
            return
        self.builder.run_solvate(
            base, custom_command=self.custom_cmds.get("solvate")
        )

    def do_genions(self, arg):
        """
        Run genions with optional custom command override. Handles three cases:
        1) Protein-only: adds ions to protein.solv.gro
        2) Protein + single ligand: adds ions to complex.solv.gro
        3) Ligand-only: adds ions to ligandX.solv.gro in current directory
        4) Lambda subdirectories: adds ions to hybrid_complex_XX.solv.gro in each lambda directory
        Usage: "genions"
        Other Options: use "set genions" to override defaults
        """

        def run_genions_and_capture(basename, custom_command=None):
            error_message = ""
            success = False
            # Patch: capture stderr/stdout
            with StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    self.builder.run_genions(basename, custom_command=custom_command)
                    output = buf.getvalue()
                    # Check for error pattern
                    if re.search(r"ERROR 1 \[file topol\\.top, line \\d+\]", output):
                        error_message = output
                        success = False
                    else:
                        success = True
                except Exception as e:
                    error_message = str(e)
                    success = False
            return success, error_message

        # Determine which system to add ions to
        if self.ligand_pdb_path and os.path.isfile("complex.gro"):
            solvated_base = "complex"
        elif not os.path.isfile("protein.gro"):
            # Ligand-only: look for ligandX.gro
            ligand_gro_files = [f"ligand{c}.gro" for c in string.ascii_uppercase]
            found = None
            for fname in ligand_gro_files:
                if os.path.isfile(fname):
                    found = fname[:-4]  # strip .gro
                    break
            if found:
                solvated_base = found
            else:
                self._log_error("No protein.gro or ligand_*.gro found for genions.")
                return
        else:
            solvated_base = "protein"
        if not self._require_pdb():
            return
        # First attempt
        success, error_message = run_genions_and_capture(
            solvated_base, custom_command=self.custom_cmds.get("genions")
        )
        if success:
            self._log_success(f"Added ions to {solvated_base}.solv.gro")
        elif "[file topol.top, line" in error_message:
            def rerun():
                return run_genions_and_capture(
                    solvated_base, custom_command=self.custom_cmds.get("genions")
                )

            self.editor.comment_out_topol_line_and_rerun_genions(rerun, error_message)
        else:
            self._log_error(f"Failed to add ions: {error_message}")

    def do_em(self, arg):
        """Run energy minimization."""
        if not self._require_pdb():
            return
        self.sim.run_em(self.basename, arg)

    def do_nvt(self, arg):
        """Run NVT equilibration."""
        if not self._require_pdb():
            return
        self.sim.run_nvt(self.basename, arg)

    def do_npt(self, arg):
        """Run NPT equilibration."""
        if not self._require_pdb():
            return
        self.sim.run_npt(self.basename, arg)

    def do_production(self, arg):
        """Run production MD simulation."""
        if not self._require_pdb():
            return
        self.sim.run_production(self.basename, arg)

    def complete_tremd_prep(self, text, line, begidx, endidx):
        """Tab completion for tremd_prep command."""
        return self._complete_filename(text, ".gro", line, begidx, endidx)

    def do_tremd_prep(self, arg):
        """Calculate temperature ladder for TREMD simulations. Usage: tremd_prep -i <init_temp> -f <final_temp> -p <prob>"""
        args = self._parse_tremd_prep(arg)
        solvated_gro = (
            "complex.solv.ions.gro"
            if os.path.exists("complex.solv.ions.gro")
            else "protein.solv.ions.gro"
        )
        if not os.path.isfile(solvated_gro):
            self._log_error(f"File not found: {solvated_gro}")
            return

        if args.i is None or args.f is None or args.p is None:
            self._log_error("Missing required flags: -i, -f, -p")

        python_exe = sys.executable
        tremd_prep_path = os.path.join(os.path.dirname(__file__), "tremd_prep.py")
        command_str = f'"{python_exe}" "{tremd_prep_path}" "{solvated_gro}" -i {args.i} -f {args.f} -p {args.p}'
        self._log_info(f"Running: {command_str}")
        success = self._execute_command(
            command=command_str,
            description="TREMD temperature ladder calculation (non-interactive)",
        )
        if not success:
            self._log_error("TREMD temperature ladder calculation failed.")

    def _parse_tremd_prep(self, arg):

        parser = argparse.ArgumentParser(
            description="Calculate Replicas Temperature Replica Exchange"
        )
        parser.add_argument(
            "-i", type=float, required=True, help="Initial Temperature (K)"
        )
        parser.add_argument(
            "-f", type=float, required=True, help="Final Temperature (K)"
        )
        parser.add_argument(
            "-p", type=float, required=True, help="Probability of Exchange (0-1)"
        )
        return parser.parse_args(arg.split())

    def do_source(self, arg):
        """Source additional .itp files into topology."""
        if not arg.strip():
            self._log_error("Usage: source <itp_file1> [itp_file2] ...")
            return

        itp_files = arg.strip().split()
        for itp_file in itp_files:
            if not os.path.exists(itp_file):
                self._log_error(f"ITP file '{itp_file}' not found.")
                continue
            self.user_itp_paths.append(os.path.abspath(itp_file))
            self._log_success(f"Added ITP file: {itp_file}")

        if self.user_itp_paths:
            self._log_info(
                "Use 'slurm' command to generate scripts with sourced ITP files."
            )

    def do_slurm(self, arg):
        """Generate SLURM scripts for MD, TREMD, or FEP simulations."""

        args = arg.strip().split()
        if len(args) < 2:
            self._log_error("Usage: slurm <sim_type> <hardware> [basename]")
            self._log_info("sim_type: 'md', 'tremd', or 'fep'")
            self._log_info("hardware: 'cpu' or 'gpu'")
            return

        sim_type = args[0].lower()
        hardware = args[1].lower()
        basename = args[2] if len(args) > 2 else self.basename

        if sim_type not in ["md", "tremd", "fep"]:
            self._log_error("sim_type must be 'md', 'tremd', or 'fep'")
            return

        if hardware not in ["cpu", "gpu"]:
            self._log_error("hardware must be 'cpu' or 'gpu'")
            return

        slurm_writer = SlurmWriter(logger=self.logger, debug=self.debug)
        slurm_writer.write_slurm_scripts(
            sim_type, hardware, basename, self.ligand_pdb_path
        )

    def print_random_quote(self):
        """Print a random quote from the quotes file."""
        try:
            # Get the path to the assets directory relative to this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(os.path.dirname(module_dir), "assets")
            quotes_path = os.path.join(assets_dir, "quotes.txt")

            with open(str(quotes_path), "r", encoding="utf-8") as f:
                quotes = f.readlines()
            if quotes:
                quote = random.choice(quotes).strip()
                if quote:
                    print(f"\n{quote}\n")
        except Exception as e:
            self._log_debug(f"Could not load quotes: {e}")

    def do_quit(self, _):
        """Exit the YAGWIP shell."""
        self._log_info("YAGWIP reminds you:")
        self.print_random_quote()
        self._log_info("Goodbye!")
        return True


def main():
    """Main entry point for YAGWIP CLI."""
    parser = argparse.ArgumentParser(description="YAGWIP - GROMACS CLI interface")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run interactive CLI"
    )
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        help="Batch process multiple PDBs using command script",
    )
    parser.add_argument(
        "-p",
        "--pdb-list",
        type=str,
        help="File containing list of PDB paths for batch processing",
    )
    parser.add_argument(
        "-d",
        "--pdb-dir",
        type=str,
        help="Directory containing PDB files for batch processing",
    )
    parser.add_argument(
        "-r", "--resume", action="store_true", help="Resume previous batch run"
    )
    parser.add_argument(
        "--ligand_builder",
        action="store_true",
        help="Use ligand builder for batch processing",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel batch processing"
    )
    parser.add_argument(
        "--workers", type=int, help="Number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--gmx-path", type=str, default="gmx", help="GROMACS executable path"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Initialize YAGWIP shell
    cli = YagwipShell(args.gmx_path)

    if args.debug:
        cli.do_debug("on")

    # Handle batch processing
    if args.batch:

        # Determine number of workers
        max_workers = args.workers
        if args.parallel and not max_workers:

            max_workers = min(mp.cpu_count(), 8)  # Auto-detect with cap

        # Initialize batch processor
        if args.parallel:
            print(
                f"Initializing parallel batch processor with {max_workers} workers..."
            )
            batch_processor = ParallelBatchProcessor(
                gmx_path=args.gmx_path,
                debug=args.debug,
                logger=cli.logger,
                ligand_builder=args.ligand_builder,
                max_workers=max_workers,
            )
        else:
            print("Initializing sequential batch processor...")
            batch_processor = ParallelBatchProcessor(
                gmx_path=args.gmx_path,
                debug=args.debug,
                logger=cli.logger,
                ligand_builder=args.ligand_builder,
                max_workers=1,  # Sequential processing
            )

        # Load PDB files
        if args.pdb_list:
            # Load from PDB list file
            batch_processor.load_pdb_list(args.pdb_list)
        elif args.pdb_dir:
            # Load from directory
            batch_processor.load_pdb_directory(args.pdb_dir)
        else:
            print(
                "[ERROR] Must specify either --pdb-list or --pdb-dir for batch processing"
            )
            sys.exit(1)

        # Execute batch
        print(f"Starting batch processing with {len(batch_processor.jobs)} jobs...")
        if args.ligand_builder:
            print("Ligand builder enabled for batch processing")
        if args.parallel:
            print(f"Parallel processing enabled with {max_workers} workers")

        results = batch_processor.execute_batch(args.batch, resume=args.resume)

        if results:
            print(
                f"Batch processing completed. Results saved in {batch_processor.results_dir}"
            )
            print(
                f"Completed: {results['completed_jobs']}/{results['total_jobs']} jobs"
            )
            print(f"Failed: {results['failed_jobs']} jobs")
            if args.parallel:
                print(
                    f"Parallel workers used: {results.get('parallel_workers', 'N/A')}"
                )
        else:
            print("Batch processing failed.")
            sys.exit(1)

    # Handle single file processing (original functionality)
    elif args.file:
        # Batch mode: read and execute commands from file
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        print(f"YAGWIP> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"[ERROR] File '{args.file}' not found.")
            sys.exit(1)

    # Interactive mode
    else:
        cli.cmdloop()


if __name__ == "__main__":
    main()
