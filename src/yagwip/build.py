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
from .base import YagwipBase
from .utils import (
    build_adjacency_matrix_fast,
    find_bonds_spatial,
)
from .log import auto_monitor

# Constants for GROMACS command inputs
PIPE_INPUTS = {"pdb2gmx": "1\n", "genion_prot": "13\n", "genion_complex": "15\n"}


class Builder(YagwipBase):
    """Handles GROMACS system building steps."""

    def __init__(self, gmx_path, debug=False, logger=None):
        """Initialize Builder."""
        super().__init__(gmx_path=gmx_path, debug=debug, logger=logger)

    @auto_monitor
    def _resolve_basename(self, basename):
        """Resolve the basename for file operations."""
        if not basename and not self.debug:
            self._log_error("No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return None
        return basename if basename else "PLACEHOLDER"

    @auto_monitor
    def run_pdb2gmx(self, basename, custom_command=None):
        """Run pdb2gmx to generate topology and coordinates."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        cmd = custom_command or (
            f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        )
        if self.debug:
            print(f"[DEBUG] Command: {cmd}")
            return
        self._log(f"Running pdb2gmx for {base}.pdb...")
        self._execute_command(cmd, f"pdb2gmx for {base}", pipe_input=PIPE_INPUTS["pdb2gmx"])

    @auto_monitor
    def run_solvate(self, basename, custom_command=None):
        """Run solvate to add solvent to the system."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        default_box = " -c -d 1.0 -bt cubic"
        default_water = "spc216.gro"
        default_cmds = [
            f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro{default_box}",
            f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs {default_water} -o {base}.solv.gro -p topol.top",
        ]
        if custom_command:
            self._log_info("Using custom solvate command")
            self._execute_command(custom_command, "custom solvate")
        else:
            for i, cmd in enumerate(default_cmds):
                self._execute_command(cmd, f"solvate step {i+1}")

    @auto_monitor
    def run_genions(self, basename, custom_command=None, fep_mode=False):
        """Run genion to add ions to the system. If lambda directories are present, copy and patch ions_fep.mdp in each lambda dir with correct lambda index and run genions in each."""
        # Detect lambda directories (case 3)
        if fep_mode:
            # FEP mode: copy and patch ions_fep.mdp in each lambda dir and run genions in each
            vdw_lambdas = [
                "0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45",
                "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"
            ]
            template_mdp = files("yagwip.templates").joinpath("ions_fep.mdp")
            # Use only the directory name for lambda value and filenames
            cwd = os.path.basename(os.getcwd())  # e.g., 'lambda_0.00'
            lam_value = cwd.replace('lambda_', '')  # '0.00'
            lambda_index = lam_value
            if lam_value in vdw_lambdas:
                lambda_index = vdw_lambdas.index(lam_value)
            # Copy ions_fep.mdp into current dir (overwrite if exists)
            dest_mdp = os.path.join(os.getcwd(), "ions_fep.mdp")
            shutil.copy2(str(template_mdp), dest_mdp)
            with open(dest_mdp, "r", encoding="utf-8") as f:
                content = f.read().replace("__LAMBDA__", str(lambda_index))
            with open(dest_mdp, "w", encoding="utf-8") as f:
                f.write(content)
            self._log_info(f"Copied and patched ions_fep.mdp in {os.getcwd()} with lambda index {lambda_index}")
            # Run genions in lambda dir
            base = f"hybrid_complex_{lam_value}"
            input_gro = f"{base}.solv.gro"
            output_gro = f"{base}.solv.ions.gro"
            tpr_out = "ions_fep.tpr"
            ion_options = "-pname NA -nname CL -conc 0.150 -neutral"
            grompp_opts = ""
            ion_pipe_input = (
                PIPE_INPUTS["genion_prot"]
                if base.endswith("protein")
                else PIPE_INPUTS["genion_complex"]
            )
            default_cmds = [
                f"{self.gmx_path} grompp -f ions_fep.mdp -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts} -maxwarn 50",
                f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}",
            ]
            if custom_command:
                self._log_info("Using custom genion command")
                self._execute_command(custom_command, "custom genion")
            else:
                for i, cmd in enumerate(default_cmds):
                    pipe_input = ion_pipe_input if i == 1 else None
                    self._execute_command(cmd, f"genion step {i+1}", pipe_input=pipe_input)
            return
        # Non-FEP or no lambda dirs: original logic
        base = self._resolve_basename(basename)
        if base is None:
            return
        else:
            mdp_file = files("yagwip.templates").joinpath("ions.mdp")
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
            f"{self.gmx_path} grompp -f {mdp_file} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts} -maxwarn 50",
            f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}",
        ]
        self._log_info(f"Running genion for {base}")
        if custom_command:
            self._log_info("Using custom genion command")
            self._execute_command(custom_command, "custom genion")
        else:
            for i, cmd in enumerate(default_cmds):
                pipe_input = ion_pipe_input if i == 1 else None
                self._execute_command(cmd, f"genion step {i+1}", pipe_input=pipe_input)


class Modeller(YagwipBase):
    """Protein structure modeller for missing residues."""

    def __init__(self, pdb, logger=None, debug=False, output_file="protein_test.pdb"):
        """Initialize Modeller."""
        super().__init__(debug=debug, logger=logger)
        self.pdb = pdb
        self.output_file = output_file

    @auto_monitor
    def find_missing_residues(self):
        """Identifies missing internal residues by checking for gaps in residue numbering."""
        self._log_info(f"Checking for missing residues in {self.pdb}")
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
        if gaps:
            self._log_warning(f"Found missing residue ranges: {gaps}")
        else:
            self._log_info("No gaps found.")
        return gaps

