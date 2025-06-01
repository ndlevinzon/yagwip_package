from .utils import run_gromacs_command
from importlib.resources import files


def run_pdb2gmx(gmx_path, basename, custom_command=None, debug=False):
    if not basename and not debug:
        print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
        return

    base = basename if basename else "PLACEHOLDER"

    default_cmd = f"{gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
    command = custom_command or default_cmd

    print(f"Running pdb2gmx for {base}.pdb...")
    run_gromacs_command(command, pipe_input="7\n", debug=debug)


def run_solvate(gmx_path, basename, custom_command=None, debug=False, arg=""):
    if not basename and not debug:
        print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
        return

    base = basename if basename else "PLACEHOLDER"

    default_box = " -c -d 1.0 -bt cubic"
    default_water = "spc216.gro"
    parts = arg.strip().split()
    box_options = parts[0] if len(parts) > 0 else default_box
    water_model = parts[1] if len(parts) > 1 else default_water

    default_cmds = [
        f"{gmx_path} editconf -f {base}.gro -o {base}.newbox.gro{box_options}",
        f"{gmx_path} solvate -cp {base}.newbox.gro -cs {water_model} -o {base}.solv.gro -p topol.top"
    ]

    if custom_command:
        print("[CUSTOM] Using custom solvate command")
        run_gromacs_command(custom_command, debug=debug)
    else:
        for cmd in default_cmds:
            run_gromacs_command(cmd, debug=debug)


def run_genions(gmx_path, basename, custom_command=None, debug=False):
    if not basename and not debug:
        print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
        return

    base = basename if basename else "PLACEHOLDER"

    if custom_command:
        print("[CUSTOM] Using custom genion command")
        run_gromacs_command(custom_command, pipe_input="13\n", debug=debug)
        return

    default_ions = files("yagwip.templates").joinpath("ions.mdp")
    input_gro = f"{base}.solv.gro"
    output_gro = f"{base}.solv.ions.gro"
    tpr_out = "ions.tpr"
    ion_options = "-pname NA -nname CL -conc 0.100 -neutral"
    grompp_opts = ""

    grompp_cmd = f"{gmx_path} grompp -f {default_ions} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts}"
    genion_cmd = f"{gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}"

    print(f"Running genion for {base}...")
    run_gromacs_command(grompp_cmd, debug=debug)
    run_gromacs_command(genion_cmd, pipe_input="13\n", debug=debug)

