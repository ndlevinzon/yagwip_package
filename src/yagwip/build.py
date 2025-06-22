from .utils import run_gromacs_command
from importlib.resources import files

# Constants for GROMACS command inputs
PIPE_INPUTS = {
    'pdb2gmx': '1\n',
    'genion_prot': '13\n',
    'genion_complex': '15\n'
}


class Builder:
    def __init__(self, gmx_path, debug=False, logger=None):
        self.gmx_path = gmx_path
        self.debug = debug
        self.logger = logger

    def _resolve_basename(self, basename):
        if not basename and not self.debug:
            msg = "[!] No PDB loaded. Use `loadPDB <filename.pdb>` first."
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)
            return None
        return basename if basename else "PLACEHOLDER"

    def run_pdb2gmx(self, basename, custom_command=None):
        base = self._resolve_basename(basename)
        if base is None:
            return

        command = custom_command or (
            f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        )

        if self.debug:
            print(f"[DEBUG] Command: {command}")
            return

        print(f"[#] Running pdb2gmx for {base}.pdb...")
        run_gromacs_command(command, pipe_input=PIPE_INPUTS['pdb2gmx'], debug=self.debug, logger=self.logger)

    def run_solvate(self, basename, arg="", custom_command=None):
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
            f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs {water_model} -o {base}.solv.gro -p topol.top"
        ]

        if self.debug:
            for cmd in default_cmds:
                print(f"[DEBUG] Command: {cmd}")
            return

        if custom_command:
            print("[CUSTOM] Using custom solvate command")
            run_gromacs_command(custom_command, debug=self.debug, logger=self.logger)
        else:
            for cmd in default_cmds:
                run_gromacs_command(cmd, debug=self.debug, logger=self.logger)

    def run_genions(self, basename, custom_command=None):
        base = self._resolve_basename(basename)
        if base is None:
            return

        default_ions = files("yagwip.templates").joinpath("ions.mdp")
        input_gro = f"{base}.solv.gro"
        output_gro = f"{base}.solv.ions.gro"
        tpr_out = "ions.tpr"
        ion_options = "-pname NA -nname CL -conc 0.150 -neutral"
        grompp_opts = ""
        ion_pipe_input = PIPE_INPUTS['genion_prot'] if base.endswith('protein') else PIPE_INPUTS['genion_complex']

        default_cmds = [
            f"{self.gmx_path} grompp -f {default_ions} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts} -maxwarn 50",
            f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}"
        ]

        print(f"[#] Running genion for {base}...")
        if self.debug:
            for cmd in default_cmds:
                print(f"[DEBUG] Command: {cmd}")
            return

        if custom_command:
            print("[CUSTOM] Using custom genion command")
            run_gromacs_command(custom_command, debug=self.debug, logger=self.logger)
        else:
            for cmd in default_cmds:
                run_gromacs_command(cmd, pipe_input=ion_pipe_input, debug=self.debug, logger=self.logger)
