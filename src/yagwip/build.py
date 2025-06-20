from .utils import run_gromacs_command
from importlib.resources import files

# Constants for GROMACS command inputs
PIPE_INPUTS = {
    'pdb2gmx': '1\n',
    'genion_prot': '13\n',
    'genion_complex': '15\n'
}


def resolve_basename(basename, debug, logger=None):
    """Helps resolve the basename for PDB files, ensuring it is set correctly."""
    if not basename and not debug:
        msg = "[!] No PDB loaded. Use `loadPDB <filename.pdb>` first."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None
    return basename if basename else "PLACEHOLDER"


def run_pdb2gmx(gmx_path, basename, custom_command=None, debug=False, logger=None):
    """
    Runs the GROMACS `pdb2gmx` command to convert a PDB file to a GROMACS-compatible .gro file
    and generate the topology. Assumes default water model (SPC/E) and ignores hydrogens.

    Parameters:
        gmx_path (str): Path to the GROMACS executable.
        basename (str): Base filename (without extension) of the PDB structure.
        custom_command (str, optional): Custom pdb2gmx command to override the default.
        debug (bool): If True, only prints the command without executing it.
        logger (logging.Logger, optional): Logger instance to capture output and status.
    """

    # If no basename is provided and not in debug mode, warn the user and exit
    base = resolve_basename(basename, debug, logger)
    if base is None:
        return

    # Use provided basename or placeholder if in debug mode
    base = basename if basename else "PLACEHOLDER"

    # Construct the default pdb2gmx command if no custom one is given
    default_cmd = f"{gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
    command = custom_command or default_cmd

    if debug:
        # If in debug mode, print the command instead of executing it
        print(f"[DEBUG] Command: {command}")
        return

    print(f"[#] Running pdb2gmx for {base}.pdb...")

    # Run the command, sending "7\n" as input (ff14SB on IRIC patched GROMACS in /levinzon/)
    run_gromacs_command(command, pipe_input=PIPE_INPUTS['pdb2gmx'], debug=debug, logger=logger)


def run_solvate(gmx_path, basename, arg="", custom_command=None, debug=False, logger=None):
    """
    Runs the GROMACS solvation workflow. This includes box setup (editconf) and
    solvent addition (solvate) around the protein/structure.

    Parameters:
        gmx_path (str): Path to GROMACS command-line tools.
        basename (str): Base name of the input file (e.g., without .gro extension).
        custom_command (str, optional): If provided, runs this custom command instead of default solvation.
        debug (bool): If True, does not execute commands, only prints them.
        arg (str): Optional CLI arguments for box type/size and solvent model
         (e.g., "-c -d 1.0 -bt triclinic spc216.gro").
        logger (Logger, optional): Logger instance for writing logs instead of printing directly.
    """

    # Ensure that a basename is provided unless in debug mode
    base = resolve_basename(basename, debug, logger)
    if base is None:
        return

    # Set default box configuration and solvent type
    default_box = " -c -d 1.0 -bt cubic"  # Center molecule, 1.0 nm buffer, cubic box
    default_water = "spc216.gro"  # Default water model file
    parts = arg.strip().split()

    # Override box or water model if specified in CLI arguments
    box_options = parts[0] if len(parts) > 0 else default_box
    water_model = parts[1] if len(parts) > 1 else default_water

    default_cmds = [
        f"{gmx_path} editconf -f {base}.gro -o {base}.newbox.gro{box_options}",
        f"{gmx_path} solvate -cp {base}.newbox.gro -cs {water_model} -o {base}.solv.gro -p topol.top"
    ]

    if debug:
        print(f"[DEBUG] Command: {default_cmds[0]}")
        print(f"[DEBUG] Command: {default_cmds[1]}")
        return
    # Define default commands for box generation and solvation
    if custom_command:
        print("[CUSTOM] Using custom solvate command")
        run_gromacs_command(custom_command, debug=debug, logger=logger)
    else:
        # Otherwise, run the default sequence of editconf and solvate
        for cmd in default_cmds:
            run_gromacs_command(cmd, debug=debug, logger=logger)


def run_genions(gmx_path, basename, custom_command=None, debug=False, logger=None):
    """
    Run GROMACS genion workflow to add counter-ions (e.g., Na+, Cl-) to a solvated system.

    Parameters:
        gmx_path (str): Path to GROMACS executables.
        basename (str): Base name of the system (used to locate solvated .gro file).
        custom_command (str, optional): If provided, this custom command overrides the default genion workflow.
        debug (bool): If True, prints commands without executing.
        logger (Logger, optional): Logger to capture command output (if provided).
    """

    # Check that a PDB structure has been loaded unless running in debug mode
    base = resolve_basename(basename, debug, logger)
    if base is None: return

    # Path to default ions.mdp configuration file bundled in templates
    default_ions = files("yagwip.templates").joinpath("ions.mdp")

    # File naming conventions for genion input/output
    input_gro = f"{base}.solv.gro"  # Input: solvated structure
    output_gro = f"{base}.solv.ions.gro"  # Output: solvated + ionized structure
    tpr_out = "ions.tpr"  # Temporary run input file for genion

    # Genion-specific options
    ion_options = "-pname NA -nname CL -conc 0.150 -neutral"  # Add Na+ and Cl- to neutralize at 0.15 M
    grompp_opts = ""  # Placeholder for additional grompp options if needed

    # If just a protein, pipe 13. If a complex, pipe 15.
    ion_pipe_input = PIPE_INPUTS['genion_prot'] if basename.endswith('protein') else PIPE_INPUTS['genion_complex']

    # Construct GROMACS commands
    default_cmds = [
        (f"{gmx_path} grompp -f {default_ions} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} "
         f"{grompp_opts} -maxwarn 15"),
        f"{gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}"
    ]

    # Execute the commands (or print if debug)
    print(f"[#] Running genion for {base}...")
    if debug:
        print(f"[DEBUG] Command: {default_cmds[0]}")
        print(f"[DEBUG] Command: {default_cmds[1]}")
        return
        # Define default commands for box generation and solvation
    if custom_command:
        print("[CUSTOM] Using custom genion command")
        run_gromacs_command(custom_command, debug=debug, logger=logger)
    else:
        # Otherwise, run the default sequence of editconf and solvate
        for cmd in default_cmds:
            run_gromacs_command(cmd, pipe_input=ion_pipe_input, debug=debug, logger=logger)
