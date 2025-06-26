import os
from .utils import run_gromacs_command, tremd_temperature_ladder, count_residues_in_gro
from importlib.resources import files


class Sim:
    def __init__(self, gmx_path, debug=False, logger=None):
        self.gmx_path = gmx_path
        self.debug = debug
        self.logger = logger

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def run_em(self, basename, arg=""):
        self._run_stage(basename, arg, default_mdp="em.mdp", suffix=".solv.ions", tprname="em")

    def run_nvt(self, basename, arg=""):
        self._run_stage(basename, arg, default_mdp="nvt.mdp", suffix=".em", tprname="nvt")

    def run_npt(self, basename, arg=""):
        self._run_stage(basename, arg, default_mdp="npt.mdp", suffix=".nvt", tprname="npt")

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

        self._execute(grompp_cmd)
        self._execute(mdrun_cmd)

    def run_tremd(self, basename, arg=""):
        args = arg.strip().split()
        if len(args) != 2 or args[0].lower() != "calc":
            print("Usage: tremd calc <filename.gro>")
            return

        gro_path = os.path.abspath(args[1])
        if not os.path.isfile(gro_path):
            print(f"[#] File not found: {gro_path}")
            return

        try:
            protein_residues, water_residues = count_residues_in_gro(gro_path)
            print(f"[#] Found {protein_residues} protein residues and {water_residues} water residues.")
        except Exception as e:
            print(f"[!] Failed to parse .gro file: {e}")
            return

        try:
            Tlow = float(input("Initial Temperature (K): "))
            Thigh = float(input("Final Temperature (K): "))
            Pdes = float(input("Exchange Probability (0 < P < 1): "))
        except ValueError:
            print("[!] Invalid numeric input.")
            return

        if not (0 < Pdes < 1) or Thigh <= Tlow:
            print("[!] Invalid parameters.")
            return

        try:
            temperatures = tremd_temperature_ladder(Tlow, Thigh, Pdes, water_residues, protein_residues, Hff=0, Vs=0, PC=1, WC=0, Tol=0.0005)
            if self.debug:
                for i, temp in enumerate(temperatures):
                    print(f"Replica {i + 1}: {temp:.2f} K")
            else:
                with open("TREMD_temp_ranges.txt", 'w') as f:
                    for i, temp in enumerate(temperatures):
                        f.write(f"Replica {i + 1}: {temp:.2f} K\n")
                print("[#] Temperature ladder saved to TREMD_temp_ranges.txt")
        except Exception as e:
            print(f"[!] Temperature calculation failed: {e}")

    def _run_stage(self, basename, arg, default_mdp, suffix, tprname):
        base = basename if basename else "PLACEHOLDER"
        self._log(f"[#] Running stage for {base} using {default_mdp}...")

        parts = arg.strip().split(maxsplit=3)
        mdpfile = parts[0] if len(parts) > 0 else str(files("yagwip.templates").joinpath(default_mdp))
        suffix = parts[1] if len(parts) > 1 else suffix
        tprname = parts[2] if len(parts) > 2 else tprname
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        input_gro = f"{base}{suffix}.gro"
        tpr_file = f"{tprname}.tpr"

        grompp_cmd = f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

        self._execute(grompp_cmd)
        self._execute(mdrun_cmd)

    def _execute(self, command):
        if self.debug:
            print(f"[RUNNING] {command}`")
            print("[DEBUG MODE] Command not executed.")
        else:
            run_gromacs_command(command, debug=self.debug, logger=self.logger)
