"""
slurm_utils.py -- Utilities for generating and customizing SLURM job scripts for YAGWIP
"""
# === Standard Library Imports ===
import os
import re
import shutil
from importlib.resources import files

# === Local Imports ===
from .base import YagwipBase


class SlurmWriter(YagwipBase):
    """
    Handles generation and writing of SLURM job scripts for GROMACS and TREMD workflows.
    """

    def __init__(self, template_pkg="yagwip.templates", logger=None, debug=False):
        super().__init__(debug=debug, logger=logger)
        self.template_dir = files(template_pkg)

    def write_slurm_scripts(self, sim_type, hardware, basename, ligand_pdb_path=None):
        """
        Generate and write SLURM scripts for the given simulation type and hardware.
        Args:
            sim_type (str): 'md', 'tremd', or 'fep'
            hardware (str): 'cpu' or 'gpu'
            basename (str): Project base name for substitution
            ligand_pdb_path (str or None): If present, use complex.solv.ions, else protein.solv.ions
        """
        if sim_type == "fep":
            # For FEP, process each lambda directory
            lambda_dirs = [d for d in os.listdir('.') if d.startswith('lambda_') and os.path.isdir(d)]
            mdp_templates = ["em_fep.mdp", "nvt_fep.mdp", "npt_fep.mdp", "production_fep.mdp"]
            for lam_dir in sorted(lambda_dirs):
                lam_value = lam_dir.replace('lambda_', '')
                # Copy and patch .mdp files
                for mdp_name in mdp_templates:
                    src_path = self.template_dir / mdp_name
                    if not src_path.is_file():
                        self._log_warning(f"Template {mdp_name} not found in {self.template_dir}")
                        continue
                    with open(str(src_path), "r", encoding="utf-8") as f:
                        content = f.read().replace("__LAMBDA__", lam_value)
                    dest_path = os.path.join(lam_dir, mdp_name)
                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    self._log_success(f"Wrote {mdp_name} to {lam_dir} with lambda {lam_value}")
                # If hardware is cpu, also copy SLURM scripts to each lambda directory
                if hardware == "cpu":
                    slurm_files = ["run_gmx_fep_min_cpu.slurm", "run_gmx_fep_cpu.slurm"]
                    for slurm_name in slurm_files:
                        src_slurm = self.template_dir / slurm_name
                        if not src_slurm.is_file():
                            self._log_warning(f"SLURM template {slurm_name} not found in {self.template_dir}")
                            continue
                        dest_slurm = os.path.join(lam_dir, slurm_name)
                        with open(str(src_slurm), "r", encoding="utf-8") as f:
                            content = f.read()
                        # Patch minimization script with correct INIT string
                        if slurm_name == "run_gmx_fep_min_cpu.slurm":
                            content = content.replace('init="__INIT__.solv.ions"', f'init="hybrid_complex_{lam_value}.solv.ions"')
                        with open(dest_slurm, "w", encoding="utf-8") as f:
                            f.write(content)
                        self._log_success(f"Copied {slurm_name} to {lam_dir} for FEP CPU workflow (lambda {lam_value})")
            return

        # Copy only relevant .mdp files
        exclude = "production.mdp" if sim_type == "tremd" else "remd_template.mdp"
        for f in self.template_dir.iterdir():
            if f.name.endswith(".mdp") and f.name != exclude:
                shutil.copy(str(f), os.getcwd())
        self._log_info(f"Copied .mdp templates for {sim_type} (excluded: {exclude})")

        # Copy minimization SLURM file for tremd
        if sim_type == "tremd":
            min_slurm = self.template_dir / "run_gmx_md_min_cpu.slurm"
            if min_slurm.is_file():
                try:
                    with open(str(min_slurm), "r", encoding="utf-8") as f:
                        min_content = f.read()
                    # Replace BASE variable in SLURM script with basename
                    min_content = re.sub(
                        r"__BASE__", basename or "PLACEHOLDER", min_content
                    )
                    min_content = re.sub(
                        r"__INIT__", "complex" or "PLACEHOLDER", min_content
                    )
                    out_min_slurm = "run_gmx_md_min_cpu.slurm"
                    with open(out_min_slurm, "w", encoding="utf-8") as f:
                        f.write(min_content)
                    self._log_success(
                        f"Customized SLURM script written: {out_min_slurm}"
                    )
                except (OSError, IOError) as e:
                    self._log_error(f"Failed to configure SLURM script: {e}")
            else:
                self._log_warning(
                    "run_gmx_md_min_cpu.slurm not found in template directory."
                )

        # Main SLURM template
        slurm_tpl_name = f"run_gmx_{sim_type}_{hardware}.slurm"
        slurm_tpl_path = self.template_dir / slurm_tpl_name
        if not slurm_tpl_path.is_file():
            self._log_error(f"SLURM template not found: {slurm_tpl_name}")
            return

        try:
            with open(str(slurm_tpl_path), "r", encoding="utf-8") as f:
                slurm_content = f.read()
            # Replace BASE variable in SLURM script with basename
            slurm_content = re.sub(
                r"__BASE__", basename or "PLACEHOLDER", slurm_content
            )
            # Write modified SLURM script
            out_slurm = f"{slurm_tpl_name}"
            with open(out_slurm, "w", encoding="utf-8") as f:
                f.write(slurm_content)
            self._log_success(f"Customized SLURM script written: {out_slurm}")
        except Exception as e:
            self._log_error(f"Failed to configure SLURM script: {e}")
