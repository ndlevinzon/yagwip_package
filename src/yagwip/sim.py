# === Standard Library Imports ===
import os
from importlib.resources import files

# === Local Imports ===
from .base import YagwipBase


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
