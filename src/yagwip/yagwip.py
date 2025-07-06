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
import shutil
import random
import argparse
import importlib.metadata
from pathlib import Path
from importlib.resources import files

# === Third-Party Imports ===
import pandas as pd

# === Local Imports ===
from .build import Builder, Modeller, LigandPipeline
from .sim import Sim
from .base import YagwipBase
from .utils import Editor, validate_gromacs_installation, complete_filename
from .slurm_writer import SlurmWriter

# === Metadata ===
__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YagwipShell(cmd.Cmd, YagwipBase):
    """
    Interactive shell for YAGWIP: Yet Another GROMACS Wrapper In Python.
    Provides a command-line interface for molecular simulation workflows.
    """

    # Intro message and prompt for the interactive CLI
    intro = f"Welcome to YAGWIP v{__version__}. Type help to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        """Initialize the YAGWIP shell with GROMACS path."""
        super().__init__(gmx_path=gmx_path, debug=False)
        self.current_pdb_path = None  # Full path to the loaded PDB file
        self.ligand_pdb_path = None  # Full path to the ligand PDB file, if any
        self.basename = None  # Base PDB filename (without extension)
        self.print_banner()  # Prints intro banner to command line
        self.user_itp_paths = []  # Stores user input paths for do_source
        self.editor = Editor()  # Initialize the file Editor class from utils.py
        self.ligand_pipeline = LigandPipeline(logger=self.logger, debug=self.debug)
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
            self._log_error(f"GROMACS Validation Error: {e}")
            self._log_error(
                "YAGWIP cannot start without GROMACS. Please install GROMACS and try again."
            )
            sys.exit(1)
        # Dictionary of custom command overrides set by the user
        self.custom_cmds = {k: "" for k in ("pdb2gmx", "solvate", "genions")}

    def _require_pdb(self):
        """Check if a PDB file is loaded."""
        if not self.current_pdb_path and not self.debug:
            self._log_error("No PDB loaded.")
            return False
        return True

    def default(self, line):
        """Throws error when command is not recognized."""
        self._log_error(f"Unknown command: {line}")

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
        from .log import setup_logger
        self.logger = setup_logger(debug_mode=self.debug)
        self._log_info(f"Debug mode is now {'ON' if self.debug else 'OFF'}")

    def print_banner(self):
        """Prints YAGWIP Banner Logo on Start."""
        try:
            banner_path = files("yagwip.assets").joinpath("yagwip_banner.txt")
            with open(str(banner_path), "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            self._log_error(f"Could not load banner: {e}")

    def do_show(self, arg):
        """Show current custom or default commands."""
        for k in ["pdb2gmx", "solvate", "genions"]:
            cmd_str = self.custom_cmds.get(k)
            self._log_info(f"{k}: {cmd_str if cmd_str else '[DEFAULT]'}")

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
            self._log_error(f"Usage: set <{'|'.join(valid_keys)}>")
            return
        # Get the default command string
        base = self.basename if self.basename else "PLACEHOLDER"
        if cmd_key == "pdb2gmx":
            default = (
                f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
            )
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
        self._log_info(f"Current command for {cmd_key}:\n{current}")
        self._log_info(
            "Type new command or press ENTER to keep current. Type 'quit' to cancel."
        )
        new_cmd = input("New command: ").strip()
        if new_cmd.lower() == "quit":
            self._log_info("Edit canceled.")
            return
        if not new_cmd:
            self.custom_cmds[cmd_key] = current
            self._log_info("Keeping existing command.")
            return
        self.custom_cmds[cmd_key] = new_cmd
        self._log_success(f"Updated command for {cmd_key}.")

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
        # Parse arguments
        parser = argparse.ArgumentParser(description="Load PDB file")
        parser.add_argument("pdb_file", help="PDB file to load")
        parser.add_argument("--ligand_builder", action="store_true", help="Use ligand builder")
        parser.add_argument("--c", type=int, default=0, help="Total charge for QM input")
        parser.add_argument("--m", type=int, default=1, help="Multiplicity for QM input")

        try:
            args = parser.parse_args(arg.split())
        except SystemExit:
            return

        pdb_file = args.pdb_file
        if not os.path.exists(pdb_file):
            self._log_error(f"PDB file '{pdb_file}' not found.")
            return

        self.current_pdb_path = os.path.abspath(pdb_file)
        self.basename = self._get_file_basename(pdb_file)
        self._log_success(f"Loaded PDB file: {pdb_file}")

        # Handle ligand building if requested
        if args.ligand_builder:
            ligand_pdb = f"{self.basename}_ligand.pdb"
            if os.path.exists(ligand_pdb):
                self.ligand_pdb_path = os.path.abspath(ligand_pdb)
                self._log_info(f"Found ligand PDB: {ligand_pdb}")

                # Check if ligand.itp exists
                if not os.path.exists("ligand.itp"):
                    self._log_info("ligand.itp not found. Running ligand building pipeline...")
                    success = self.ligand_pipeline.convert_pdb_to_mol2(ligand_pdb)
                    if success:
                        self._log_success("Ligand building pipeline completed successfully.")
                    else:
                        self._log_error("Ligand building pipeline failed.")
                else:
                    self._log_info("ligand.itp already exists. Skipping ligand building.")
            else:
                self._log_warning(f"Ligand PDB '{ligand_pdb}' not found. Skipping ligand building.")

    def do_pdb2gmx(self, arg):
        """Run pdb2gmx to generate topology and coordinates."""
        if not self._require_pdb():
            return
        custom_cmd = self.custom_cmds.get("pdb2gmx")
        self.builder.run_pdb2gmx(self.basename, custom_cmd)

    def do_solvate(self, arg):
        """Run solvate to add solvent to the system."""
        if not self._require_pdb():
            return
        custom_cmd = self.custom_cmds.get("solvate")
        self.builder.run_solvate(self.basename, arg, custom_cmd)

    def do_genions(self, arg):
        """Run genion to add ions to the system."""
        if not self._require_pdb():
            return
        custom_cmd = self.custom_cmds.get("genions")
        self.builder.run_genions(self.basename, custom_cmd)

    def do_em(self, arg):
        """Run energy minimization."""
        if not self._require_pdb():
            return
        self.sim.run_em(self.basename, arg)

    def do_nvt(self, arg):
        """Run NVT equilibration."""
        if not self._require_pdb():
            return
        self.sim.run_nvt(self.basename, arg)

    def do_npt(self, arg):
        """Run NPT equilibration."""
        if not self._require_pdb():
            return
        self.sim.run_npt(self.basename, arg)

    def do_production(self, arg):
        """Run production MD simulation."""
        if not self._require_pdb():
            return
        self.sim.run_production(self.basename, arg)

    def complete_tremd(self, text, line, begidx, endidx):
        """Tab completion for tremd command."""
        return complete_filename(text, ".gro", line, begidx, endidx)

    def do_tremd(self, arg):
        """Calculate temperature ladder for TREMD simulations."""
        if not self._require_pdb():
            return
        self.sim.run_tremd(self.basename, arg)

    def do_source(self, arg):
        """Source additional .itp files into topology."""
        if not arg.strip():
            self._log_error("Usage: source <itp_file1> [itp_file2] ...")
            return

        itp_files = arg.strip().split()
        for itp_file in itp_files:
            if not os.path.exists(itp_file):
                self._log_error(f"ITP file '{itp_file}' not found.")
                continue
            self.user_itp_paths.append(os.path.abspath(itp_file))
            self._log_success(f"Added ITP file: {itp_file}")

        if self.user_itp_paths:
            self._log_info("Use 'slurm' command to generate scripts with sourced ITP files.")

    def do_slurm(self, arg):
        """Generate SLURM scripts for MD or TREMD simulations."""
        if not self._require_pdb():
            return

        args = arg.strip().split()
        if len(args) < 2:
            self._log_error("Usage: slurm <sim_type> <hardware> [basename]")
            self._log_info("sim_type: 'md' or 'tremd'")
            self._log_info("hardware: 'cpu' or 'gpu'")
            return

        sim_type = args[0].lower()
        hardware = args[1].lower()
        basename = args[2] if len(args) > 2 else self.basename

        if sim_type not in ['md', 'tremd']:
            self._log_error("sim_type must be 'md' or 'tremd'")
            return

        if hardware not in ['cpu', 'gpu']:
            self._log_error("hardware must be 'cpu' or 'gpu'")
            return

        slurm_writer = SlurmWriter(logger=self.logger, debug=self.debug)
        slurm_writer.write_slurm_scripts(sim_type, hardware, basename, self.ligand_pdb_path)

    def print_random_quote(self):
        """Print a random quote from the quotes file."""
        try:
            quotes_path = files("yagwip.assets").joinpath("quotes.txt")
            with open(str(quotes_path), "r", encoding="utf-8") as f:
                quotes = f.readlines()
            if quotes:
                quote = random.choice(quotes).strip()
                if quote:
                    print(f"\n{quote}\n")
        except Exception as e:
            self._log_debug(f"Could not load quotes: {e}")

    def do_quit(self, _):
        """Exit the YAGWIP shell."""
        self.print_random_quote()
        self._log_info("Goodbye!")
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
