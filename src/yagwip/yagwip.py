"""
yagwip.py: (Y)et (A)nother (G)ROMACS (W)rapper (I)n (P)ython

Portions copyright (c) 2025 the Authors.
Authors: Nathan Levinzon, Olivier Mailhot

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# === Standard Library Imports ===
import io
import os
import cmd
import sys
import shlex
import shutil
import random
import argparse
import importlib.metadata
from importlib.resources import files

# === Third-Party Imports ===
import pandas as pd

# === Local Imports ===
from .build import Builder, Modeller, LigandPipeline
from .sim import Sim
from .utils import Editor, LoggingMixin, setup_logger, validate_gromacs_installation, complete_filename, ToolChecker
from .slurm_writer import SlurmWriter

# === Metadata ===
__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YagwipShell(cmd.Cmd, LoggingMixin):
    """
    Interactive shell for YAGWIP: Yet Another GROMACS Wrapper In Python.
    Provides a command-line interface for molecular simulation workflows.
    """
    # Intro message and prompt for the interactive CLI
    intro = f"Welcome to YAGWIP v{__version__}. Type help to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        """Initialize the YAGWIP shell with GROMACS path."""
        super().__init__()
        self.debug = False  # Toggle debug mode
        self.gmx_path = gmx_path  # Path to GROMACS executable (e.g., "gmx")
        self.logger = setup_logger(debug_mode=self.debug)  # Initialize logging
        self.current_pdb_path = None  # Full path to the loaded PDB file
        self.ligand_pdb_path = None  # Full path to the ligand PDB file, if any
        self.basename = None  # Base PDB filename (without extension)
        self.print_banner()  # Prints intro banner to command line
        self.user_itp_paths = []  # Stores user input paths for do_source
        self.editor = Editor()  # Initialize the file Editor class from utils.py
        # Initialize the Editor class from utils.py
        self.modeller = Modeller(
            pdb="protein.pdb", debug=self.debug, logger=self.logger
        )
        # Initialize the Sim class from sim.py
        self.sim = Sim(gmx_path=self.gmx_path, debug=self.debug, logger=self.logger)

        # Initialize the Builder and Sim classes from build.py and sim.py
        self.builder = Builder(
            gmx_path=self.gmx_path, debug=self.debug, logger=self.logger
        )
        # Validate GROMACS installation
        try:
            validate_gromacs_installation(gmx_path)
        except RuntimeError as e:
            print(f"[ERROR] GROMACS Validation Error: {e}")
            print("\nYAGWIP cannot start without GROMACS. Please install GROMACS and try again.")
            sys.exit(1)
        # Dictionary of custom command overrides set by the user
        self.custom_cmds = {k: "" for k in ("pdb2gmx", "solvate", "genions")}

    def _require_pdb(self):
        """Check if a PDB file is loaded."""
        if not self.current_pdb_path and not self.debug:
            self._log("[ERROR] No PDB loaded.")
            return False
        return True

    def default(self, line):
        """Throws error when command is not recognized."""
        self._log(f"[ERROR] Unknown command: {line}")

    def do_debug(self, arg):
        """
        Debug Mode: Simply prints commands to the command line that
        would have otherwise be executed. Prints to console instead of log

        Usage: Toggle with 'debug', 'debug on', or 'debug off'"
        """
        arg = arg.lower().strip()
        if arg == "on":
            self.debug = True
        elif arg == "off":
            self.debug = False
        else:
            self.debug = not self.debug
        # Update logger and simulation mode
        self.logger = setup_logger(debug_mode=self.debug)
        self._log(f"[DEBUG] Debug mode is now {'ON' if self.debug else 'OFF'}")

    def print_banner(self):
        """Prints YAGWIP Banner Logo on Start."""
        try:
            banner_path = files("yagwip.assets").joinpath("yagwip_banner.txt")
            with open(str(banner_path), "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            self._log(f"[ERROR] Could not load banner: {e}")

    def do_show(self, arg):
        """Show current custom or default commands."""
        for k in ["pdb2gmx", "solvate", "genions"]:
            cmd_str = self.custom_cmds.get(k)
            self._log(f"{k}: {cmd_str if cmd_str else '[DEFAULT]'}")

    def do_set(self, arg):
        """
        Edit the default command string for pdb2gmx, solvate, or genions.
        Usage:
            set pdb2gmx
            set solvate
            set genions
        The user is shown the current command and can modify it inline.
        Press ENTER to accept the modified command.
        Type 'quit' to cancel.
        """
        valid_keys = ["pdb2gmx", "solvate", "genions"]
        cmd_key = arg.strip().lower()
        if cmd_key not in valid_keys:
            print(f"Usage: set <{'|'.join(valid_keys)}>")
            return
        # Get the default command string
        base = self.basename if self.basename else "PLACEHOLDER"
        if cmd_key == "pdb2gmx":
            default = f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        elif cmd_key == "solvate":
            default = (
                f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro -c -d 1.0 -bt cubic && "
                f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs spc216.gro -o {base}.solv.gro -p topol.top"
            )
        elif cmd_key == "genions":
            ions_mdp = "ions.mdp"  # assuming it's copied to current dir already
            default = (
                f"{self.gmx_path} grompp -f {ions_mdp} -c {base}.solv.gro -r {base}.solv.gro -p topol.top -o ions.tpr && "
                f"{self.gmx_path} genion -s ions.tpr -o {base}.solv.ions.gro -p topol.top -pname NA -nname CL -conc 0.150 -neutral"
            )
        # Show current command and prompt for new input
        current = self.custom_cmds.get(cmd_key) or default
        self._log(f"[EDIT {cmd_key}] Current command:\n{current}")
        self._log("Type new command or press ENTER to keep current. Type 'quit' to cancel.")
        new_cmd = input("New command: ").strip()
        if new_cmd.lower() == "quit":
            self._log("[SET] Edit canceled.")
            return
        if not new_cmd:
            self.custom_cmds[cmd_key] = current
            self._log("[SET] Keeping existing command.")
            return
        self.custom_cmds[cmd_key] = new_cmd
        self._log(f"[SET] Updated command for {cmd_key}.")

    def complete_loadpdb(self, text, line=None, begidx=None, endidx=None):
        """Tab completion for .pdb files."""
        return complete_filename(text, ".pdb", line, begidx, endidx)

    def do_loadpdb(self, arg):
        """
        Usage: "loadpdb X.pdb [--ligand_builder] [--c CHARGE] [--m MULTIPLICITY] (Requires ORCA)."
                --ligand_builder: Run the ligand building pipeline if ligand.itp is missing.
                --c: Set the total charge for QM input (default 0)
                --m: Set the multiplicity for QM input (default 1)
        """
        args = shlex.split(arg)
        if not args:
            print("Usage: loadpdb <filename.pdb> [--ligand_builder] [--c CHARGE] [--m MULTIPLICITY]")
            return
        filename = args[0]
        use_ligand_builder = "--ligand_builder" in args
        charge = int(args[args.index("--c") + 1]) if "--c" in args else 0
        multiplicity = int(args[args.index("--m") + 1]) if "--m" in args else 1
        full_path = os.path.abspath(filename)
        if not os.path.isfile(full_path):
            self._log(f"[ERROR] '{filename}' not found.")
            return
        # Store the full path and basename for later use in the build pipeline
        self.current_pdb_path = full_path
        self.basename = os.path.splitext(os.path.basename(full_path))[0]
        self._log(f"PDB file loaded: {full_path}")
        # Read all lines from the PDB file
        with open(full_path, "r") as f:
            lines = f.readlines()
        # Extract all lines representing heteroatoms (typically ligands or cofactors)
        hetatm_lines = [line for line in lines if line.startswith("HETATM")]
        # Always rewrite the protein portion with HIS substitutions
        protein_file = "protein.pdb"
        if hetatm_lines:
            # If ligand atoms were found, prepare a separate ligand file
            ligand_file = "ligand.pdb"
            self.ligand_pdb_path = os.path.abspath(ligand_file)
            # Open output files for writing protein and ligand portions
            with open(protein_file, "w", encoding="utf-8") as prot_out, open(ligand_file, "w", encoding="utf-8") as lig_out:
                for line in lines:
                    if line.startswith("HETATM"):
                        # Replace ligand residue name with LIG
                        lig_out.write(line[:17] + "LIG" + line[20:])
                    else:
                        # Replace HSP or HSD with HIS in protein
                        if line[17:20] in ("HSP", "HSD"):
                            line = line[:17] + "HIS" + line[20:]
                        prot_out.write(line)
            self._log(f"Detected ligand. Split into: {protein_file}, {ligand_file}")
            # Determine if the ligand contains hydrogen atoms (important for parameterization)
            has_hydrogens = any(line[76:78].strip() == "H" or line[12:16].strip().startswith("H") for line in hetatm_lines)
            if not has_hydrogens:
                self._log("[WARNING] Ligand appears to lack hydrogen atoms. Consider checking hydrogens and valences.")
            # Check that the ligand.itp file exists and preprocess it if so
            if os.path.isfile("ligand.itp"):
                self._log("Checking ligand.itp...")
                self.editor.append_ligand_atomtypes_to_forcefield()
                self.editor.modify_improper_dihedrals_in_ligand_itp()
                self.editor.rename_residue_in_itp_atoms_section()
            else:
                self._log("ligand.itp not found in the current directory.")
                if use_ligand_builder:
                    # Copy amber14sb.ff directory from templates to current working directory
                    self._log("Setting up force field files for ligand building...")
                    amber_ff_source = files("yagwip.templates").joinpath("amber14sb.ff")
                    amber_ff_dest = "amber14sb.ff"

                    if os.path.exists(amber_ff_dest):
                        self._log(f"[INFO] {amber_ff_dest} already exists in current directory.")
                    else:
                        try:
                            shutil.copytree(amber_ff_source, amber_ff_dest)
                            self._log(f"[SUCCESS] Copied {amber_ff_source} to {amber_ff_dest}")
                        except Exception as e:
                            self._log(f"[ERROR] Failed to copy amber14sb.ff: {e}")
                            return
                    ligand_pipeline = LigandPipeline(logger=self.logger, debug=self.debug)
                    ligand_pdb = "ligand.pdb"
                    mol2_file = ligand_pipeline.convert_pdb_to_mol2(ligand_pdb)
                    if mol2_file is None:
                        self._log("[ERROR] MOL2 generation failed. Aborting ligand pipeline...")
                        return
                    # Find the start and end of the ATOM section
                    with open(mol2_file, encoding="utf-8") as f:
                        lines = f.readlines()
                    atom_start = atom_end = None
                    for i, line in enumerate(lines):
                        if line.strip() == "@<TRIPOS>ATOM":
                            atom_start = i + 1
                        elif line.strip().startswith("@<TRIPOS>BOND") and atom_start is not None:
                            atom_end = i
                            break
                    if atom_start is None:
                        self._log("[ERROR] Could not find ATOM section in MOL2 file.")
                        return
                    if atom_end is None:
                        atom_end = len(lines)
                    atom_lines = lines[atom_start:atom_end]
                    # Parse atom lines into DataFrame
                    df_atoms = pd.read_csv(
                        io.StringIO("".join(atom_lines)),
                        sep=r"\s+",
                        header=None,
                        names=[
                            "atom_id",
                            "atom_name",
                            "x",
                            "y",
                            "z",
                            "atom_type",
                            "subst_id",
                            "subst_name",
                            "charge",
                            "status_bit",
                        ],
                    )
                    # Generate ORCA Geometry Optimization input
                    orca_geom_input = mol2_file.replace(".mol2", ".inp")
                    ligand_pipeline.mol2_dataframe_to_orca_charge_input(
                        df_atoms,
                        orca_geom_input,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                    # Run ORCA Geometry Optimization
                    ligand_pipeline.run_orca(orca_geom_input)
                    # Append atom charges to mol2
                    ligand_pipeline.apply_orca_charges_to_mol2(mol2_file, "orca/ligand.property.txt")
                    ligand_pipeline.run_parmchk2(mol2_file)  # creates ligand.frcmod
                    ligand_pipeline.run_acpype(mol2_file)  # convert to gromacs
                    return
                self._log("[ERROR] ligand.itp not found.")
                return
        else:
            # If no HETATM lines are found, treat entire file as protein
            self.ligand_pdb_path = None
            with open(protein_file, "w", encoding="utf-8") as prot_out:
                for line in lines:
                    # Normalize histidine variants to 'HIS'
                    if line[17:20] in ("HSP", "HSD"):
                        line = line[:17] + "HIS" + line[20:]
                    prot_out.write(line)
            self._log(
                "No HETATM entries found. Wrote corrected PDB to protein.pdb and using it as apo protein."
            )
        self.modeller.find_missing_residues()

    def do_pdb2gmx(self, arg):
        """
        Run pdb2gmx. If ligand is present, treat protein and ligand separately.
        Usage: "pdb2gmx"
        """
        if not self._require_pdb():
            return
        protein_pdb = "protein"
        output_gro = f"{protein_pdb}.gro"
        self.builder.run_pdb2gmx(protein_pdb, custom_command=self.custom_cmds["pdb2gmx"])
        if not os.path.isfile(output_gro):
            self._log(f"[ERROR] Expected {output_gro} was not created.")
            return
        # Combine ligand coordinates
        if self.ligand_pdb_path and os.path.getsize("ligand.pdb") > 0:
            self.editor.append_ligand_coordinates_to_gro(output_gro, "ligand.pdb", "complex.gro")
            self.editor.include_ligand_itp_in_topol("topol.top", "LIG")
        else:
            shutil.copy(str(output_gro), "complex.gro")  # only protein

    def do_solvate(self, arg):
        """
        Run solvate with optional custom command override. This command should be run after pdb2gmx.
        Usage: "solvate"
        Other Options: use "set solvate" to override defaults
        """

        complex_pdb = "complex" if self.ligand_pdb_path else "protein"
        if not self._require_pdb():
            return
        self.builder.run_solvate(complex_pdb, custom_command=self.custom_cmds["solvate"])

    def do_genions(self, arg):
        """
        Run genions with optional custom command override. This command should be run after solvate.
        Usage: "genions"
        Other Options: use "set genions" to override defaults
        """
        solvated_pdb = "complex" if self.ligand_pdb_path else "protein"
        if not self._require_pdb():
            return
        self.builder.run_genions(solvated_pdb, custom_command=self.custom_cmds["genions"])

    def do_em(self, arg):
        """
        Runs default energy minimization on the command line
        Usage: "em"
        """
        if not self._require_pdb():
            return
        self.sim.run_em(self.basename, arg=arg)

    def do_nvt(self, arg):
        """
        Runs default NVT equilibration on the command line
        Usage: "em"
        """
        if not self._require_pdb():
            return
        self.sim.run_nvt(self.basename, arg=arg)

    def do_npt(self, arg):
        """
        Runs default NPT equilibration on the command line
        Usage: "npt"
        """
        if not self._require_pdb():
            return
        self.sim.run_npt(self.basename, arg=arg)

    def do_production(self, arg):
        """
        Runs default production-phase MD on the command line
        Usage: "production"
        """
        if not self._require_pdb():
            return
        self.sim.run_production(self.basename, arg=arg)

    def complete_tremd(self, text, line, begidx, endidx):
        """Adds tab completion for .solv.ions.gro for use in TREMD replica calculations"""
        args = line.strip().split()
        if len(args) >= 2 and args[1] == "calc":
            return complete_filename(text, "solv.ions.gro", line, begidx, endidx)
        return []

    def do_tremd(self, arg):
        """
        Generate a TREMD temperature ladder based on a user-specified .gro file.
        This computes replica exchange temperature ranges using the van der Spoel predictor.

        Usage: "tremd calc X.solv.ions.gro"
        """
        self.sim.run_tremd(self.basename, arg=arg)

    def do_source(self, arg):
        """
        Add a custom .itp include to be added to all topol.top files.
        This replaces any existing #include lines not in the user-defined list.

        Usage: source /absolute/path/to/custom.itp
        """
        itp_path = arg.strip()
        if not itp_path.endswith(".itp"):
            self._log("[ERROR] Must provide a path to a .itp file.")
            return
        if not os.path.isfile(itp_path):
            self._log(f"[ERROR] File '{itp_path}' not found.")
            return

        # Add new path to list (no duplicates)
        if itp_path not in self.user_itp_paths:
            self.user_itp_paths.append(itp_path)
            self._log(f"Added custom .itp include: {itp_path}")
        else:
            self._log(f"Path already in include list: {itp_path}")
        # Apply all includes to all topol.top files
        self.editor.insert_itp_into_top_files(self.user_itp_paths, root_dir=os.getcwd())
        # Display all tracked includes
        self._log("\nCurrent custom ITP includes:")
        for p in self.user_itp_paths:
            print(f'  #include "{p}"')

    def do_slurm(self, arg):
        """
        Setup SLURM job scripts.

        Usage:
            slurm md cpu
            slurm md gpu
            slurm tremd cpu
            slurm tremd gpu

        Copies template .mdp files and SLURM script to the current directory based on options.
        """
        args = arg.strip().lower().split()
        if (
            len(args) != 2
            or args[0] not in ["md", "tremd"]
            or args[1] not in ["cpu", "gpu"]
        ):
            print("Usage: slurm <md|tremd> <cpu|gpu>")
            return
        sim_type, hardware = args
        writer = SlurmWriter(logger=self.logger)
        writer.write_slurm_scripts(
            sim_type=sim_type,
            hardware=hardware,
            basename=self.basename,
            ligand_pdb_path=self.ligand_pdb_path,
        )

    def print_random_quote(self):
        """
        Prints random quote on exit.
        Quotes: scr/yagwip/assets/quotes.txt
        """
        try:
            quote_path = files("yagwip.assets").joinpath("quotes.txt")
            with open(str(quote_path), "r", encoding="utf-8") as f:
                quotes = [line.strip() for line in f if line.strip()]
            if quotes:
                print(f"\nYAGWIP Reminds You...\n{random.choice(quotes)}\n")
        except Exception as e:
            self._log(f"([ERROR] Unable to load quotes: {e})")

    def do_quit(self, _):
        """
        Quit the CLI.
        Usage: "quit"
        """
        self.print_random_quote()
        print(f"Copyright (c) 2025 {__author__} \nQuitting YAGWIP.")
        return True


def main():
    """Main entry point for YAGWIP CLI."""
    parser = argparse.ArgumentParser(description="YAGWIP - GROMACS CLI interface")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run interactive CLI"
    )
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")
    args = parser.parse_args()
    cli = YagwipShell("gmx")
    if args.file:
        # Batch mode: read and execute commands from file
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        print(f"YAGWIP> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"[ERROR] File '{args.file}' not found.")
            sys.exit(1)
    else:
        cli.cmdloop()


if __name__ == "__main__":
    main()
