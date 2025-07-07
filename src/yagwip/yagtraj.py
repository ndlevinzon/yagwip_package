"""
yagtraj.py: Trajectory analysis tool for Gromacs MD simulations

Copyright (c) 2025 the Authors.
Authors: Nathan Levinzon, Olivier Mailhot

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import cmd
import sys
import os
from .base import YagwipBase
from .utils import *
from importlib.resources import files
import importlib.metadata
import random

__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YagtrajShell(cmd.Cmd, YagwipBase):
    intro = f"Welcome to YAGTRAJ v{__version__}. Type help to list commands."
    prompt = "YAGTRAJ> "

    def __init__(self, gmx_path):
        super().__init__(gmx_path=gmx_path, debug=False)
        self.current_tpr = None  # Current TPR file
        self.current_traj = None  # Current trajectory file
        self.print_banner()  # Prints intro banner to command line

        # Validate GROMACS installation
        try:
            validate_gromacs_installation(gmx_path)
        except RuntimeError as e:
            self._log_error(f"GROMACS Validation Error: {e}")
            self._log_error(
                "YAGWIP cannot start without GROMACS. Please install GROMACS and try again."
            )
            sys.exit(1)

    def default(self, line):
        """Throws error when command is not recognized"""
        self._log_error(f"Unknown command: {line}")

    def do_debug(self, arg):
        """
        Debug Mode: Simply prints commands to the command line that
        would have otherwise be executed. Prints to console instead of log

        Usage: Toggle with 'debug', 'debug on', or 'debug off'"
        """
        arg = arg.lower().strip()

        # Parse input to determine new debug state
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
        """
        Prints YAGTRAJ Banner Logo on Start
        Banner: src/yagwip/assets/yagtraj_banner.txt
        """
        try:
            banner_path = files("yagwip.assets").joinpath("yagtraj_banner.txt")
            with open(str(banner_path), "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            self._log_error(f"Could not load banner: {e}")

    def do_load(self, arg):
        """
        Load trajectory files for analysis.
        Usage: load <tpr_file> [traj_file]
        Example: load production.tpr production.xtc
        """
        args = arg.strip().split()
        if len(args) < 1:
            self._log_error("Usage: load <tpr_file> [traj_file]")
            return

        tpr_file = args[0]
        if not os.path.exists(tpr_file):
            self._log_error(f"TPR file '{tpr_file}' not found.")
            return

        self.current_tpr = tpr_file
        self._log_success(f"Loaded TPR file: {tpr_file}")

        if len(args) > 1:
            traj_file = args[1]
            if not os.path.exists(traj_file):
                self._log_error(f"Trajectory file '{traj_file}' not found.")
                return
            self.current_traj = traj_file
            self._log_success(f"Loaded trajectory file: {traj_file}")

    def _require_files(self):
        """Check if required files are loaded."""
        if not self.current_tpr:
            self._log_error("No TPR file loaded. Use 'load <tpr_file>' first.")
            return False
        return True

    def do_rmsd(self, arg):
        """
        Calculate RMSD for protein backbone.
        Usage: rmsd [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "rmsd.xvg"

        # Build the gmx rms command
        command = f"{self.gmx_path} rms -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Calculating RMSD for protein backbone")

        # Use run_gromacs_command from utils
        success = run_gromacs_command(
            command=command,
            pipe_input="4\n4\n",  # Select backbone for both reference and fit
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"RMSD calculation completed. Output: {output_file}")
        else:
            self._log_error("RMSD calculation failed.")

    def do_rgyr(self, arg):
        """
        Calculate radius of gyration.
        Usage: rgyr [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "rgyr.xvg"

        command = f"{self.gmx_path} gyrate -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Calculating radius of gyration")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n",  # Select protein
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(
                f"Radius of gyration calculation completed. Output: {output_file}"
            )
        else:
            self._log_error("Radius of gyration calculation failed.")

    def do_sasa(self, arg):
        """
        Calculate solvent accessible surface area.
        Usage: sasa [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "sasa.xvg"

        command = f"{self.gmx_path} sasa -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Calculating solvent accessible surface area")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n",  # Select protein
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"SASA calculation completed. Output: {output_file}")
        else:
            self._log_error("SASA calculation failed.")

    def do_hbond(self, arg):
        """
        Calculate hydrogen bonds.
        Usage: hbond [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "hbond.xvg"

        command = f"{self.gmx_path} hbond -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Calculating hydrogen bonds")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n1\n",  # Select protein for both donor and acceptor
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(
                f"Hydrogen bond calculation completed. Output: {output_file}"
            )
        else:
            self._log_error("Hydrogen bond calculation failed.")

    def do_distance(self, arg):
        """
        Calculate distance between two groups.
        Usage: distance <group1> <group2> [output_file]
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 2:
            self._log_error("Usage: distance <group1> <group2> [output_file]")
            return

        group1, group2 = args[0], args[1]
        output_file = args[2] if len(args) > 2 else "distance.xvg"

        command = f"{self.gmx_path} distance -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file} -select 'group {group1} plus group {group2}'"

        self._log_info(f"Calculating distance between groups {group1} and {group2}")

        success = run_gromacs_command(
            command=command,
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"Distance calculation completed. Output: {output_file}")
        else:
            self._log_error("Distance calculation failed.")

    def do_energy(self, arg):
        """
        Calculate potential energy.
        Usage: energy [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "energy.xvg"

        command = f"{self.gmx_path} energy -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Calculating potential energy")

        success = run_gromacs_command(
            command=command,
            pipe_input="10\n",  # Select potential energy
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"Energy calculation completed. Output: {output_file}")
        else:
            self._log_error("Energy calculation failed.")

    def do_trjconv(self, arg):
        """
        Convert trajectory format.
        Usage: trjconv <output_format> [output_file]
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 1:
            self._log_error("Usage: trjconv <output_format> [output_file]")
            return

        output_format = args[0]
        output_file = args[1] if len(args) > 1 else f"output.{output_format}"

        command = f"{self.gmx_path} trjconv -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info(f"Converting trajectory to {output_format} format")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n",  # Select protein
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"Trajectory conversion completed. Output: {output_file}")
        else:
            self._log_error("Trajectory conversion failed.")

    def do_tremd_demux(self, arg):
        """
        Demultiplex TREMD trajectory.
        Usage: tremd_demux <replica_count> [output_prefix]
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 1:
            self._log_error("Usage: tremd_demux <replica_count> [output_prefix]")
            return

        try:
            replica_count = int(args[0])
        except ValueError:
            self._log_error("Replica count must be an integer")
            return

        output_prefix = args[1] if len(args) > 1 else "demux"

        command = f"{self.gmx_path} trjcat -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -demux {output_prefix} -n {replica_count}"

        self._log_info(f"Demultiplexing TREMD trajectory for {replica_count} replicas")

        success = run_gromacs_command(
            command=command,
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(
                f"TREMD demultiplexing completed. Output prefix: {output_prefix}"
            )
        else:
            self._log_error("TREMD demultiplexing failed.")

    def do_tremd_rmsd(self, arg):
        """
        Calculate RMSD for TREMD trajectories.
        Usage: tremd_rmsd <replica_count> [output_prefix]
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 1:
            self._log_error("Usage: tremd_rmsd <replica_count> [output_prefix]")
            return

        try:
            replica_count = int(args[0])
        except ValueError:
            self._log_error("Replica count must be an integer")
            return

        output_prefix = args[1] if len(args) > 1 else "tremd_rmsd"

        self._log_info(f"Calculating RMSD for {replica_count} TREMD replicas")

        for i in range(replica_count):
            traj_file = f"demux{i+1}.xtc"
            if not os.path.exists(traj_file):
                self._log_warning(f"Trajectory file {traj_file} not found, skipping")
                continue

            output_file = f"{output_prefix}_replica{i+1}.xvg"
            command = f"{self.gmx_path} rms -s {self.current_tpr} -f {traj_file} -o {output_file}"

            success = run_gromacs_command(
                command=command,
                pipe_input="4\n4\n",  # Select backbone for both reference and fit
                debug=self.debug,
                logger=self.logger,
            )

            if success:
                self._log_success(
                    f"RMSD calculation completed for replica {i+1}. Output: {output_file}"
                )
            else:
                self._log_error(f"RMSD calculation failed for replica {i+1}")

    def do_tremd_rmsf(self, arg):
        """
        Calculate RMSF for TREMD trajectories.
        Usage: tremd_rmsf <replica_count> [output_prefix]
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 1:
            self._log_error("Usage: tremd_rmsf <replica_count> [output_prefix]")
            return

        try:
            replica_count = int(args[0])
        except ValueError:
            self._log_error("Replica count must be an integer")
            return

        output_prefix = args[1] if len(args) > 1 else "tremd_rmsf"

        self._log_info(f"Calculating RMSF for {replica_count} TREMD replicas")

        for i in range(replica_count):
            traj_file = f"demux{i+1}.xtc"
            if not os.path.exists(traj_file):
                self._log_warning(f"Trajectory file {traj_file} not found, skipping")
                continue

            output_file = f"{output_prefix}_replica{i+1}.xvg"
            command = f"{self.gmx_path} rmsf -s {self.current_tpr} -f {traj_file} -o {output_file}"

            success = run_gromacs_command(
                command=command,
                pipe_input="4\n",  # Select backbone
                debug=self.debug,
                logger=self.logger,
            )

            if success:
                self._log_success(
                    f"RMSF calculation completed for replica {i+1}. Output: {output_file}"
                )
            else:
                self._log_error(f"RMSF calculation failed for replica {i+1}")

    def do_tremd_pca(self, arg):
        """
        Perform PCA analysis on TREMD trajectories.
        Usage: tremd_pca <replica_count> [output_prefix]
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 1:
            self._log_error("Usage: tremd_pca <replica_count> [output_prefix]")
            return

        try:
            replica_count = int(args[0])
        except ValueError:
            self._log_error("Replica count must be an integer")
            return

        output_prefix = args[1] if len(args) > 1 else "tremd_pca"

        self._log_info(f"Performing PCA analysis for {replica_count} TREMD replicas")

        for i in range(replica_count):
            traj_file = f"demux{i+1}.xtc"
            if not os.path.exists(traj_file):
                self._log_warning(f"Trajectory file {traj_file} not found, skipping")
                continue

            # Covariance matrix
            covar_file = f"{output_prefix}_replica{i+1}_covar.xvg"
            command = f"{self.gmx_path} covar -s {self.current_tpr} -f {traj_file} -o {covar_file}"

            success = run_gromacs_command(
                command=command,
                pipe_input="4\n4\n",  # Select backbone for both reference and fit
                debug=self.debug,
                logger=self.logger,
            )

            if success:
                self._log_success(
                    f"PCA analysis completed for replica {i+1}. Output: {covar_file}"
                )
            else:
                self._log_error(f"PCA analysis failed for replica {i+1}")

    def do_tremd_temp(self, arg):
        """
        Analyze temperature exchange in TREMD simulations.
        Usage: tremd_temp [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "tremd_temp.xvg"

        command = f"{self.gmx_path} energy -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Analyzing temperature exchange in TREMD simulation")

        success = run_gromacs_command(
            command=command,
            pipe_input="16\n",  # Select temperature
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"Temperature analysis completed. Output: {output_file}")
        else:
            self._log_error("Temperature analysis failed.")

    def do_tremd_energy(self, arg):
        """
        Analyze energy exchange in TREMD simulations.
        Usage: tremd_energy [output_file]
        """
        if not self._require_files():
            return

        output_file = arg.strip() if arg.strip() else "tremd_energy.xvg"

        command = f"{self.gmx_path} energy -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log_info("Analyzing energy exchange in TREMD simulation")

        success = run_gromacs_command(
            command=command,
            pipe_input="10\n",  # Select potential energy
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log_success(f"Energy analysis completed. Output: {output_file}")
        else:
            self._log_error("Energy analysis failed.")

    def do_info(self, arg):
        """Show information about loaded files."""
        if self.current_tpr:
            self._log_info(f"TPR file: {self.current_tpr}")
        if self.current_traj:
            self._log_info(f"Trajectory file: {self.current_traj}")
        if not self.current_tpr and not self.current_traj:
            self._log_info("No files loaded. Use 'load <tpr_file>' to load files.")

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
        """Exit the YAGTRAJ shell."""
        self.print_random_quote()
        self._log_info("Goodbye!")
        return True


def main():
    parser = argparse.ArgumentParser(description="YAGTRAJ - GROMACS MD Analysis")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run interactive CLI"
    )
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")

    args = parser.parse_args()
    cli = YagtrajShell("gmx")

    if args.file:
        # Batch mode: read and execute commands from file
        try:
            with open(args.file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(
                        "#"
                    ):  # skip empty lines and comments
                        print(f"YAGTRAJ> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"[!] File '{args.file}' not found.")
            sys.exit(1)
    else:
        # Interactive mode
        cli.cmdloop()


if __name__ == "__main__":
    main()
