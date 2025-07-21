"""
gromacs_runner.py: GROMACS and ligand building utilities for YAGWIP.
"""

# === Standard Library Imports ===
import os
import shutil
from importlib.resources import files

# === Local Imports ===
from yagwip.base import YagwipBase
from utils.log_utils import auto_monitor

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
        self._execute_command(
            cmd, f"pdb2gmx for {base}", pipe_input=PIPE_INPUTS["pdb2gmx"]
        )

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
                "0.00",
                "0.05",
                "0.10",
                "0.15",
                "0.20",
                "0.25",
                "0.30",
                "0.35",
                "0.40",
                "0.45",
                "0.50",
                "0.55",
                "0.60",
                "0.65",
                "0.70",
                "0.75",
                "0.80",
                "0.85",
                "0.90",
                "0.95",
                "1.00",
            ]
            module_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(os.path.dirname(module_dir), "templates")
            template_mdp = os.path.join(assets_dir, "ions_fep.mdp")

            # Use only the directory name for lambda value and filenames
            cwd = os.path.basename(os.getcwd())  # e.g., 'lambda_0.00'
            lam_value = cwd.replace("lambda_", "")  # '0.00'
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
            self._log_info(
                f"Copied and patched ions_fep.mdp in {os.getcwd()} with lambda index {lambda_index}"
            )
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
                    self._execute_command(
                        cmd, f"genion step {i+1}", pipe_input=pipe_input
                    )
            return
        # Non-FEP or no lambda dirs: original logic
        base = self._resolve_basename(basename)
        if base is None:
            return
        else:
            mdp_file = files("templates").joinpath("ions.mdp")
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


class Sim(YagwipBase):
    def __init__(self, gmx_path, debug=False, logger=None):
        super().__init__(gmx_path=gmx_path, debug=debug, logger=logger)

    def run_em(self, basename, arg=""):
        self._run_stage(
            basename, arg, default_mdp="em.mdp", suffix=".solv.ions", tprname="em"
        )

    def run_nvt(self, basename, arg=""):
        self._run_stage(
            basename, arg, default_mdp="nvt.mdp", suffix=".em", tprname="nvt"
        )

    def run_npt(self, basename, arg=""):
        self._run_stage(
            basename, arg, default_mdp="npt.mdp", suffix=".nvt", tprname="npt"
        )

    def run_production(self, basename, arg=""):
        parts = arg.strip().split(maxsplit=3)
        default_mdp = files("yagwip.templates").joinpath("production.mdp")
        mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
        inputname = parts[1] if len(parts) > 1 else "npt."
        outname = parts[2] if len(parts) > 2 else "md1ns"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        input_gro = f"{inputname}gro"
        tpr_file = f"{outname}.tpr"

        grompp_cmd = f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {outname} {mdrun_suffix}"

        self._execute_command(grompp_cmd, "grompp for production")
        self._execute_command(mdrun_cmd, "mdrun for production")

    def _run_stage(self, basename, arg, default_mdp, suffix, tprname):
        base = basename if basename else "PLACEHOLDER"
        self._log_info(f"Running stage for {base} using {default_mdp}")

        parts = arg.strip().split(maxsplit=3)
        mdpfile = (
            parts[0]
            if len(parts) > 0
            else str(files("yagwip.templates").joinpath(default_mdp))
        )
        suffix = parts[1] if len(parts) > 1 else suffix
        tprname = parts[2] if len(parts) > 2 else tprname
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        input_gro = f"{base}{suffix}.gro"
        tpr_file = f"{tprname}.tpr"

        grompp_cmd = f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

        self._execute_command(grompp_cmd, f"grompp for {tprname}")
        self._execute_command(mdrun_cmd, f"mdrun for {tprname}")
