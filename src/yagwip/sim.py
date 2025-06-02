import os
from .utils import run_gromacs_command, tremd_temperature_ladder, count_residues_in_gro
from importlib.resources import files


def run_em(gmx_path, basename, arg="", debug=False, logger=None):
    base = basename if basename else "PLACEHOLDER"
    print(f"Running energy minimization for {base}...")

    parts = arg.strip().split(maxsplit=3)
    default_mdp = files("yagwip.templates").joinpath("em.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
    suffix = parts[1] if len(parts) > 1 else ".solv.ions"
    tprname = parts[2] if len(parts) > 2 else "em"
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    input_gro = f"{base}{suffix}.gro"
    tpr_file = f"{tprname}.tpr"

    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_nvt(gmx_path, basename, arg="", debug=False, logger=None):
    base = basename if basename else "PLACEHOLDER"
    print(f"Running NVT equilibration for {base}...")

    parts = arg.strip().split(maxsplit=3)
    default_mdp = files("yagwip.templates").joinpath("nvt.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
    suffix = parts[1] if len(parts) > 1 else ".em"
    tprname = parts[2] if len(parts) > 2 else "nvt"
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    input_gro = f"{base}{suffix}.gro"
    tpr_file = f"{tprname}.tpr"

    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_npt(gmx_path, basename, arg="", debug=False, logger=None):
    base = basename if basename else "PLACEHOLDER"
    print(f"Running NPT equilibration for {base}...")

    parts = arg.strip().split(maxsplit=3)
    default_mdp = files("yagwip.templates").joinpath("npt.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
    suffix = parts[1] if len(parts) > 1 else ".nvt"
    tprname = parts[2] if len(parts) > 2 else "npt"
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    input_gro = f"{base}{suffix}.gro"
    tpr_file = f"{tprname}.tpr"

    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_production(gmx_path, basename, arg="", debug=False, logger=None):
    base = basename if basename else "PLACEHOLDER"
    print(f"Running production MD for {base}...")

    parts = arg.strip().split(maxsplit=3)
    default_mdp = files("yagwip.templates").joinpath("production.mdp")
    mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
    inputname = parts[1] if len(parts) > 1 else "npt."
    outname = parts[2] if len(parts) > 2 else "md1ns"
    mdrun_suffix = parts[3] if len(parts) > 3 else ""

    input_gro = f"{inputname}gro"
    tpr_file = f"{outname}.tpr"

    grompp_cmd = (
        f"{gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
    )
    mdrun_cmd = f"{gmx_path} mdrun -v -deffnm {outname} {mdrun_suffix}"

    if debug:
        print(f"[RUNNING] {grompp_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
        print(f"[RUNNING] {mdrun_cmd}")
        print(f"[DEBUG MODE] Command not executed.")
    else:
        run_gromacs_command(grompp_cmd, debug=debug, logger=logger)
        run_gromacs_command(mdrun_cmd, debug=debug, logger=logger)


def run_tremd(gmx_path, basename, arg="", debug=False):
    args = arg.strip().split()
    if len(args) != 2 or args[0].lower() != "calc":
        print("Usage: tremd calc <filename.gro>")
        return

    gro_path = os.path.abspath(args[1])
    if not os.path.isfile(gro_path):
        print(f"[ERROR] File not found: {gro_path}")
        return

    try:
        protein_residues, water_residues = count_residues_in_gro(gro_path)
        print(f"[INFO] Found {protein_residues} protein residues and {water_residues} water residues.")
    except Exception as e:
        print(f"[ERROR] Failed to parse .gro file: {e}")
        return

    try:
        Tlow = float(input("Initial Temperature (K): "))
        Thigh = float(input("Final Temperature (K): "))
        Pdes = float(input("Exchange Probability (0 < P < 1): "))
    except ValueError:
        print("[ERROR] Invalid numeric input.")
        return

    if not (0 < Pdes < 1):
        print("[ERROR] Exchange probability must be between 0 and 1.")
        return
    if Thigh <= Tlow:
        print("[ERROR] Final temperature must be greater than initial temperature.")
        return

    try:
        temperatures = tremd_temperature_ladder(
            Tlow=Tlow,
            Thigh=Thigh,
            Pdes=Pdes,
            Nw=water_residues,
            Np=protein_residues,
            Hff=0,
            Vs=0,
            PC=1,
            WC=0,
            Tol=0.0005
        )

        if debug:
            print("[DEBUG MODE] TREMD temperature ladder:")
            for i, temp in enumerate(temperatures):
                print(f"Replica {i + 1}: {temp:.2f} K")
        else:
            out_file = "TREMD_temp_ranges.txt"
            with open(out_file, 'w') as f:
                f.write("# TREMD Temperature Ladder\n")
                for i, temp in enumerate(temperatures):
                    f.write(f"Replica {i + 1}: {temp:.2f} K\n")
            print(f"[TREMD] Temperature ladder saved to {out_file}")

    except Exception as e:
        print(f"[ERROR] Temperature calculation failed: {e}")
