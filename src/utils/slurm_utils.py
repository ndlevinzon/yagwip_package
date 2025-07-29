"""
slurm_utils.py -- Utilities for generating and customizing SLURM job scripts for YAGWIP

Handles three simulation types with two hardware options each:
- MD simulations: CPU and GPU
- TREMD simulations: CPU and GPU
- FEP simulations: CPU and GPU
"""

# === Standard Library Imports ===
import os
import re
import shutil
from importlib.resources import files

# === Local Imports ===
from yagwip.base import YagwipBase


class SlurmWriter(YagwipBase):
    """
    Handles generation and writing of SLURM job scripts for GROMACS workflows.

    Supports three simulation types:
    1. MD (Molecular Dynamics) - regular protein/ligand simulations
    2. TREMD (Temperature Replica Exchange MD) - temperature ladder simulations
    3. FEP (Free Energy Perturbation) - lambda window simulations

    Each type supports both CPU and GPU hardware configurations.
    """

    def __init__(self, template_pkg="templates", logger=None, debug=False):
        super().__init__(debug=debug, logger=logger)
        # Use importlib.resources to access templates from the package
        self.template_dir = files("templates")

    def write_slurm_scripts(self, sim_type, hardware, basename, ligand_pdb_path=None):
        """
        Generate and write SLURM scripts for the given simulation type and hardware.

        Args:
            sim_type (str): 'md', 'tremd', or 'fep'
            hardware (str): 'cpu' or 'gpu'
            basename (str): Project base name for substitution
            ligand_pdb_path (str or None): If present, use complex.solv.ions, else protein.solv.ions
        """
        # Validate inputs
        if sim_type not in ['md', 'tremd', 'fep']:
            self._log_error(f"Invalid simulation type: {sim_type}. Must be 'md', 'tremd', or 'fep'")
            return

        if hardware not in ['cpu', 'gpu']:
            self._log_error(f"Invalid hardware type: {hardware}. Must be 'cpu' or 'gpu'")
            return

        self._log_info(f"Generating SLURM scripts for {sim_type.upper()} simulation on {hardware.upper()}")

        # Route to appropriate handler based on simulation type
        if sim_type == "md":
            self._handle_md_slurm(hardware, basename, ligand_pdb_path)
        elif sim_type == "tremd":
            self._handle_tremd_slurm(hardware, basename)
        elif sim_type == "fep":
            self._handle_fep_slurm(hardware, basename)
        else:
            self._log_error(f"Unknown simulation type: {sim_type}")

    def _handle_md_slurm(self, hardware, basename, ligand_pdb_path=None):
        """
        Handle MD (Molecular Dynamics) SLURM script generation.

        Args:
            hardware (str): 'cpu' or 'gpu'
            basename (str): Project base name
            ligand_pdb_path (str or None): Determines if this is a complex or protein-only simulation
        """
        self._log_info("Processing MD simulation SLURM scripts")

        # Copy MD-specific MDP files (exclude TREMD-specific files and FEP-specific files)
        self._copy_mdp_files(exclude_files=["production_remd.mdp"], exclude_patterns=["*_fep.mdp"])

        # Determine initialization file name
        init_name = "complex" if ligand_pdb_path else "protein"

        # Copy and customize SLURM scripts
        if hardware == "cpu":
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_md_min_cpu.slurm",
                output_name="run_gmx_md_min_cpu.slurm",
                basename=basename,
                init_name=init_name
            )
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_md_cpu.slurm",
                output_name="run_gmx_md_cpu.slurm",
                basename=basename,
                init_name=init_name
            )
        elif hardware == "gpu":
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_md_min_gpu.slurm",
                output_name="run_gmx_md_min_gpu.slurm",
                basename=basename,
                init_name=init_name
            )
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_md_gpu.slurm",
                output_name="run_gmx_md_gpu.slurm",
                basename=basename,
                init_name=init_name
            )

    def _handle_tremd_slurm(self, hardware, basename, ligand_pdb_path=None):
        """
        Handle TREMD (Temperature Replica Exchange MD) SLURM script generation.

        Args:
            hardware (str): 'cpu' or 'gpu'
            basename (str): Project base name
        """
        self._log_info("Processing TREMD simulation SLURM scripts")

        # Copy TREMD-specific MDP files (exclude regular production files and FEP-specific files)
        self._copy_mdp_files(exclude_files=["production.mdp"], exclude_patterns=["*_fep.mdp"])

        # Determine initialization file name
        init_name = "complex" if ligand_pdb_path else "protein"

        # Copy and customize SLURM scripts
        if hardware == "cpu":
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_tremd_min_cpu.slurm",
                output_name="run_gmx_tremd_min_cpu.slurm",
                basename=basename,
                init_name=init_name
            )
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_tremd_cpu.slurm",
                output_name="run_gmx_tremd_cpu.slurm",
                basename=basename,
                init_name=init_name
            )
        elif hardware == "gpu":
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_tremd_min_gpu.slurm",
                output_name="run_gmx_tremd_min_gpu.slurm",
                basename=basename,
                init_name=init_name
            )
            self._copy_and_customize_slurm_script(
                template_name="run_gmx_tremd_gpu.slurm",
                output_name="run_gmx_tremd_gpu.slurm",
                basename=basename,
                init_name=init_name
            )

    def _handle_fep_slurm(self, hardware, basename):
        """
        Handle FEP (Free Energy Perturbation) SLURM script generation.

        Args:
            hardware (str): 'cpu' or 'gpu'
            basename (str): Project base name
        """
        self._log_info("Processing FEP simulation SLURM scripts")

        # Step 1: Prepare FEP directories using Editor
        try:
            from utils.pipeline_utils import Editor
            editor = Editor()
            editor.prepare_fep_directories()
            self._log_success("FEP directory preparation completed")
        except Exception as e:
            self._log_error(f"Failed to prepare FEP directories: {e}")
            return

        # Step 2: Copy FEP SLURM scripts to current working directory
        if hardware == "cpu":
            self._copy_fep_slurm_scripts_cwd("cpu")
        elif hardware == "gpu":
            self._copy_fep_slurm_scripts_cwd("gpu")
        else:
            self._log_error(f"Unsupported hardware type for FEP: {hardware}")

    def _copy_fep_slurm_scripts_cwd(self, hardware):
        """
        Copy FEP SLURM scripts to current working directory.

        Args:
            hardware (str): 'cpu' or 'gpu'
        """
        slurm_templates = [
            f"run_gmx_fep_min_{hardware}.slurm",
            f"run_gmx_fep_{hardware}.slurm"
        ]

        for slurm_name in slurm_templates:
            src_slurm = self.template_dir / slurm_name

            if not src_slurm.is_file():
                self._log_warning(f"FEP SLURM template {slurm_name} not found in {self.template_dir}")
                continue

            try:
                # Copy to current working directory
                dest_slurm = os.path.join(os.getcwd(), slurm_name)
                with open(str(src_slurm), "r", encoding="utf-8") as f:
                    content = f.read()
                with open(dest_slurm, "w", encoding="utf-8") as f:
                    f.write(content)

                self._log_success(f"Copied {slurm_name} to current working directory")

            except Exception as e:
                self._log_error(f"Failed to copy FEP SLURM script {slurm_name}: {e}")

    def _copy_mdp_files(self, exclude_files=None, exclude_patterns=None):
        """
        Copy MDP files from templates, excluding specified files and patterns.

        Args:
            exclude_files (list): List of MDP files to exclude from copying
            exclude_patterns (list): List of glob patterns to exclude from copying
        """
        if exclude_files is None:
            exclude_files = []
        if exclude_patterns is None:
            exclude_patterns = []

        copied_count = 0
        for f in self.template_dir.iterdir():
            if not f.name.endswith(".mdp"):
                continue

            # Check if file should be excluded by name
            if f.name in exclude_files:
                continue

            # Check if file should be excluded by pattern
            excluded_by_pattern = False
            for pattern in exclude_patterns:
                if pattern.startswith("*") and pattern.endswith(".mdp"):
                    suffix = pattern[1:]  # Remove the "*" prefix
                    if f.name.endswith(suffix):
                        excluded_by_pattern = True
                        break
                elif pattern == f.name:
                    excluded_by_pattern = True
                    break

            if excluded_by_pattern:
                continue

            try:
                shutil.copy(str(f), os.getcwd())
                copied_count += 1
            except Exception as e:
                self._log_error(f"Failed to copy {f.name}: {e}")

        self._log_success(f"Copied {copied_count} MDP files (excluded: {exclude_files}, patterns: {exclude_patterns})")

    def _copy_and_customize_slurm_script(self, template_name, output_name, basename, init_name):
        """
        Copy and customize a SLURM script template.

        Args:
            template_name (str): Name of template file in templates directory
            output_name (str): Name of output file
            basename (str): Project base name for substitution
            init_name (str): Initialization file name for substitution
        """
        template_path = self.template_dir / template_name

        if not template_path.is_file():
            self._log_warning(f"SLURM template {template_name} not found in {self.template_dir}")
            return

        try:
            # Read template content
            with open(str(template_path), "r", encoding="utf-8") as f:
                content = f.read()

            # Replace placeholders
            content = re.sub(r"__BASE__", basename or "PLACEHOLDER", content)
            content = re.sub(r"__INIT__", init_name, content)

            # Write customized script
            with open(output_name, "w", encoding="utf-8") as f:
                f.write(content)

            self._log_success(f"Customized SLURM script written: {output_name}")

        except Exception as e:
            self._log_error(f"Failed to configure SLURM script {template_name}: {e}")

    def _copy_fep_mdp_files(self, lambda_dir, mdp_templates, lambda_index):
        """
        Copy and patch FEP MDP files for a specific lambda directory.

        Args:
            lambda_dir (str): Lambda directory path
            mdp_templates (list): List of MDP template names
            lambda_index (int): Lambda index for substitution
        """
        for mdp_name in mdp_templates:
            src_path = self.template_dir / mdp_name

            if not src_path.is_file():
                self._log_warning(f"FEP template {mdp_name} not found in {self.template_dir}")
                continue

            try:
                # Read and patch template
                with open(str(src_path), "r", encoding="utf-8") as f:
                    content = f.read().replace("__LAMBDA__", str(lambda_index))

                # Write to lambda directory
                dest_path = os.path.join(lambda_dir, mdp_name)
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self._log_success(f"Wrote {mdp_name} to {lambda_dir} with lambda index {lambda_index}")

            except Exception as e:
                self._log_error(f"Failed to copy FEP MDP file {mdp_name} to {lambda_dir}: {e}")

    def _copy_fep_slurm_scripts(self, lambda_dir, lam_value, lambda_index):
        """
        Copy FEP SLURM scripts for a specific lambda directory.

        Args:
            lambda_dir (str): Lambda directory path
            lam_value (str): Lambda value string
            lambda_index (int): Lambda index
        """
        slurm_files = [
            "run_gmx_fep_min_cpu.slurm",
            "run_gmx_fep_cpu.slurm"
        ]

        for slurm_name in slurm_files:
            src_slurm = self.template_dir / slurm_name

            if not src_slurm.is_file():
                self._log_warning(f"FEP SLURM template {slurm_name} not found in {self.template_dir}")
                continue

            try:
                # Copy to current working directory (not lambda directory, as per original logic)
                dest_slurm = os.path.join(os.getcwd(), slurm_name)
                with open(str(src_slurm), "r", encoding="utf-8") as f:
                    content = f.read()
                with open(dest_slurm, "w", encoding="utf-8") as f:
                    f.write(content)

                self._log_success(f"Copied {slurm_name} for FEP CPU workflow (lambda {lam_value}, index {lambda_index})")

            except Exception as e:
                self._log_error(f"Failed to copy FEP SLURM script {slurm_name}: {e}")
