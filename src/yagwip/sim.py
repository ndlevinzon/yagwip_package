import os
from .base import YagwipBase
from .utils import run_gromacs_command, tremd_temperature_ladder, count_residues_in_gro
from importlib.resources import files


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

    def run_tremd(self, basename, arg=""):
        args = arg.strip().split()
        if len(args) != 2 or args[0].lower() != "calc":
            self._log_error("Usage: tremd calc <filename.gro>")
            return

        gro_path = os.path.abspath(args[1])
        if not os.path.isfile(gro_path):
            self._log_error(f"File not found: {gro_path}")
            return

        try:
            protein_residues, water_residues = count_residues_in_gro(gro_path)
            self._log_info(
                f"Found {protein_residues} protein residues and {water_residues} water residues."
            )
        except Exception as e:
            self._log_error(f"Failed to parse .gro file: {e}")
            return

        try:
            Tlow = float(input("Initial Temperature (K): "))
            Thigh = float(input("Final Temperature (K): "))
            Pdes = float(input("Exchange Probability (0 < P < 1): "))
        except ValueError:
            self._log_error("Invalid numeric input.")
            return

        if not (0 < Pdes < 1) or Thigh <= Tlow:
            self._log_error("Invalid parameters.")
            return

        try:
            temperatures = tremd_temperature_ladder(
                water_residues,  # Nw: Number of water molecules
                protein_residues,  # Np: Number of protein residues
                Tlow,  # Tlow: Minimum temperature (K)
                Thigh,  # Thigh: Maximum temperature (K)
                Pdes,  # Pdes: Desired exchange probability
                WC=3,  # Water constraints (3 = all constraints)
                PC=1,  # Protein constraints (1 = H atoms only)
                Hff=0,  # Hydrogen force field switch (0 = standard)
                Vs=0,  # Volume correction (0 = no)
                Tol=0.0005,  # Tolerance for convergence
            )
            if self.debug:
                for i, temp in enumerate(temperatures):
                    self._log_debug(f"Replica {i + 1}: {temp:.2f} K")
            else:
                with open("TREMD_temp_ranges.txt", "w") as f:
                    for i, temp in enumerate(temperatures):
                        f.write(f"Replica {i + 1}: {temp:.2f} K\n")
                self._log_success("Temperature ladder saved to TREMD_temp_ranges.txt")
        except Exception as e:
            self._log_error(f"Temperature calculation failed: {e}")

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
