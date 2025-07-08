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
from .config import validate_gromacs_installation
from .utils import *
import argparse as _argparse
import shutil
import subprocess
from typing import List
import sys as _sys
from importlib.resources import files
import importlib.metadata
import random

__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YagtrajShell(cmd.Cmd, YagwipBase):
    intro = f"Welcome to YAGTRAJ v{__version__}. Type help to list commands."
    prompt = "YAGTRAJ> "

    def __init__(self, gmx_path):
        cmd.Cmd.__init__(self)
        YagwipBase.__init__(self, gmx_path=gmx_path, debug=False)
        self.current_tpr = None  # Current TPR file
        self.current_traj = None  # Current trajectory file
        self.basename = None  # Base name for loaded TPR
        self.user_itp_paths = []  # For future extensibility
        self.custom_cmds = {k: "" for k in ("rmsd", "rgyr", "sasa", "hbond", "distance", "energy", "trjconv")}  # Example for traj analysis
        self.print_banner()  # Prints intro banner to command line
        # Validate GROMACS installation
        try:
            validate_gromacs_installation(gmx_path)
        except RuntimeError as e:
            self._log_error(f"GROMACS Validation Error: {e}")
            self._log_error(
                "YAGTRAJ cannot start without GROMACS. Please install GROMACS and try again."
            )
            sys.exit(1)

    def default(self, line):
        """Throws error when command is not recognized"""
        self._log_error(f"Unknown command: {line}")

    def do_debug(self, arg):
        """
        Debug Mode: Enhanced logging with detailed resource statistics and command information.
        Commands are still executed, but with verbose output including system resources.

        Usage: Toggle with 'debug', 'debug on', or 'debug off'
        """
        arg = arg.lower().strip()
        if arg == "on":
            self.debug = True
        elif arg == "off":
            self.debug = False
        else:
            self.debug = not self.debug
        from .log import setup_logger
        self.logger = setup_logger(debug_mode=self.debug)
        if self.debug:
            self._log_info("Debug mode is now ON")
            self._log_info("Enhanced debug mode will show:")
            self._log_info("  - Detailed command information")
            self._log_info("  - System resource statistics")
            self._log_info("  - Runtime monitoring data")
            self._log_info("  - All commands will still be executed")
        else:
            self._log_info("Debug mode is now OFF")

    def do_set(self, arg):
        """
        Edit the default command string for rmsd, rgyr, sasa, hbond, distance, energy, trjconv.
        Usage:
            set rmsd
            set rgyr
            set sasa
            set hbond
            set distance
            set energy
            set trjconv
        The user is shown the current command and can modify it inline.
        Press ENTER to accept the modified command.
        Type 'quit' to cancel.
        """
        valid_keys = list(self.custom_cmds.keys())
        cmd_key = arg.strip().lower()
        if cmd_key not in valid_keys:
            self._log_error(f"Usage: set <{'|'.join(valid_keys)}>")
            return
        default = self.custom_cmds[cmd_key] or f"[DEFAULT {cmd_key.upper()} COMMAND]"
        self._log_info(f"Current command for {cmd_key}: {default}")
        new_cmd = input(f"Edit command for {cmd_key} (ENTER to keep, 'quit' to cancel): ")
        if new_cmd.strip().lower() == "quit":
            self._log_info("Edit cancelled.")
            return
        if new_cmd.strip():
            self.custom_cmds[cmd_key] = new_cmd.strip()
            self._log_success(f"Custom command for {cmd_key} set.")
        else:
            self._log_info("No changes made.")

    def do_show(self, arg):
        """
        Show current custom or default commands for trajectory analysis.
        Usage: show
        """
        for k in self.custom_cmds:
            cmd_str = self.custom_cmds.get(k)
            self._log_info(f"{k}: {cmd_str if cmd_str else '[DEFAULT]'}")

    def do_runtime(self, arg):
        """
        Show runtime statistics and performance metrics.
        Usage: runtime
        """
        if hasattr(self, "runtime_monitor"):
            summary = self.runtime_monitor.get_summary()
            if summary:
                self._log_info("=== Runtime Statistics ===")
                self._log_info(f"Total Operations: {summary['total_operations']}")
                self._log_info(f"Successful: {summary['successful_operations']}")
                self._log_info(f"Failed: {summary['failed_operations']}")
                self._log_info(f"Success Rate: {summary['success_rate']:.1%}")
                self._log_info(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
                self._log_info(f"Average Duration: {summary['average_duration_seconds']:.2f}s")
            else:
                self._log_info("No runtime data available yet.")
        else:
            self._log_info("Runtime monitoring not available.")

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
        Perform full T-REMD demultiplexing pipeline using defaults:
        - input_dir: current working directory
        - deffnm: 'remd'
        - demux_script: 'demux.pl'
        Usage: tremd_demux
        """
        import os
        input_dir = os.getcwd()
        deffnm = 'remd'
        demux_script = 'demux.pl'
        out_dir = os.path.join(input_dir, "remd_analysis_results")
        log_tmp = os.path.join(out_dir, "remd_logs")
        os.makedirs(out_dir, exist_ok=True)

        # Store for later use by analyze_replicas
        self.last_tremd_input_dir = input_dir
        self.last_tremd_deffnm = deffnm
        self.last_tremd_demux_script = demux_script

        try:
            # 1. Detect replicas
            replicas = detect_replicas(input_dir)
            self._log_info(f"Found {len(replicas)} TREMD directories: {replicas}")

            # 2. Aggregate logs
            aggregate_logs(replicas, f"{deffnm}.log", log_tmp)
            self._log_info("Aggregated logs.")

            # 3. Run demux
            run_demux(os.path.join(log_tmp, "REMD.log"), out_dir, demux_script)
            self._log_info("Ran demux.")

            # 4. Demultiplex trajectories
            xtc_files = [f"{i}/{deffnm}.xtc" for i in replicas]
            demux_trajectories(xtc_files, os.path.join(out_dir, "replica_index.xvg"))
            self._log_info("Demultiplexed trajectories.")

            self._log_success(f"T-REMD demux pipeline complete. Results in {out_dir}")
        except Exception as e:
            self._log_error(f"T-REMD demux pipeline failed: {e}")

    def do_analyze_replicas(self, arg):
        """
        Analyze each demuxed replica after demuxing has been performed.
        Usage: analyze_replicas
        """
        # Use last values from tremd_demux
        input_dir = getattr(self, "last_tremd_input_dir", None)
        deffnm = getattr(self, "last_tremd_deffnm", None)
        demux_script = getattr(self, "last_tremd_demux_script", None)
        if not input_dir or not deffnm or not demux_script:
            self._log_error("You must run tremd_demux before analyze_replicas.")
            return

        out_dir = os.path.join(input_dir, "remd_analysis_results")
        replica_index_path = os.path.join(out_dir, "replica_index.xvg")
        if not os.path.exists(replica_index_path):
            self._log_error("Demuxing has not been performed. Run tremd_demux first.")
            return

        try:
            # Detect replicas
            replicas = detect_replicas(input_dir)
            self._log_info(f"Found {len(replicas)} TREMD directories: {replicas}")

            # Analyze each replica
            for replica in replicas:
                self._log_info(f"Processing replica {replica}...")
                rep_outdir = os.path.join(out_dir, f"replica_{replica}")
                analyze_replica(replica, rep_outdir, deffnm)
                self._log_info(f"Replica {replica} analysis complete.")

            # Combine energy files and extract potential
            edr_files = [f"{i}/{deffnm}.edr" for i in replicas if os.path.exists(f"{i}/{deffnm}.edr")]
            combined_edr = os.path.join(out_dir, "combined.edr")
            if edr_files:
                combine_energies(edr_files, combined_edr)
                extract_potential(combined_edr, os.path.join(out_dir, "kbT_scalar.xvg"))
                self._log_info("Combined energy files and extracted potential.")

            self._log_success(f"Replica analysis complete. Results in {out_dir}")
        except Exception as e:
            self._log_error(f"T-REMD analysis pipeline failed: {e}")

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

# --- T-REMD Analysis Utilities ---

def detect_replicas(input_dir: str) -> List[str]:
    """Detect replica directories named as integers."""
    return sorted([d for d in os.listdir(input_dir) if d.isdigit() and os.path.isdir(os.path.join(input_dir, d))], key=int)

def aggregate_logs(replicas: List[str], log_name: str, log_tmp: str):
    os.makedirs(log_tmp, exist_ok=True)
    for i in replicas:
        src = f"{i}/{log_name}"
        dst = f"{log_tmp}/remd_{i}.log"
        if os.path.exists(src):
            shutil.copy(src, dst)
    with open(f"{log_tmp}/REMD.log", "w") as outfile:
        for i in replicas:
            log_path = f"{log_tmp}/remd_{i}.log"
            if os.path.exists(log_path):
                with open(log_path) as infile:
                    outfile.write(infile.read())

def run_demux(log_file: str, out_dir: str, demux_script: str = "demux.pl"):
    subprocess.run([demux_script, log_file], check=True)
    for fname in ["replica_index.xvg", "replica_temp.xvg"]:
        if os.path.exists(fname):
            shutil.move(fname, out_dir)

def demux_trajectories(xtc_files: List[str], index_file: str):
    cmd = ["gmx", "trjcat", "-f"] + xtc_files + ["-demux", index_file]
    subprocess.run(cmd, check=True)

def analyze_replica(replica: str, outdir: str, deffnm: str = "remd"):
    os.makedirs(outdir, exist_ok=True)
    # Move demuxed .xtc
    xtc_src = f"{replica}_trajout.xtc"
    xtc_dst = os.path.join(outdir, f"{replica}_trajout.xtc")
    if os.path.exists(xtc_src):
        shutil.move(xtc_src, xtc_dst)
    # Copy .tpr
    tpr_src = f"{replica}/{deffnm}.tpr"
    tpr_dst = os.path.join(outdir, f"demuxed_{replica}.tpr")
    if os.path.exists(tpr_src):
        shutil.copy(tpr_src, tpr_dst)
    # RMSD
    subprocess.run(f"echo 4 4 | gmx rms -s {tpr_dst} -f {xtc_dst} -o {outdir}/rmsd.xvg -res", shell=True, check=True)
    # RMSF
    subprocess.run(f"echo 4 | gmx rmsf -s {tpr_dst} -f {xtc_dst} -o {outdir}/rmsf.xvg -res", shell=True, check=True)
    # PCA
    centered = os.path.join(outdir, "traj_centered.xtc")
    centered_rt = os.path.join(outdir, "traj_centered_rot_trans.xtc")
    subprocess.run(f"echo 0 0 | gmx trjconv -s {tpr_dst} -f {xtc_dst} -o {centered} -center -pbc mol", shell=True, check=True)
    subprocess.run(f"echo 0 0 | gmx trjconv -s {tpr_dst} -f {centered} -o {centered_rt} -ur compact -fit rot+trans", shell=True, check=True)
    subprocess.run(f"echo 4 4 | gmx covar -s {tpr_dst} -f {centered_rt} -o {outdir}/eigenval.xvg -v {outdir}/eigenvec.trr", shell=True, check=True)
    subprocess.run(f"echo 4 4 | gmx anaeig -v {outdir}/eigenvec.trr -s {tpr_dst} -f {centered_rt} -proj {outdir}/proj.xvg", shell=True, check=True)
    # Temperature extraction
    edr_src = f"{replica}/{deffnm}.edr"
    edr_dst = os.path.join(outdir, "ener.edr")
    if os.path.exists(edr_src):
        shutil.copy(edr_src, edr_dst)
        subprocess.run(f"echo Temperature | gmx energy -f {edr_dst} -o {outdir}/temp.xvg", shell=True, check=True)


def combine_energies(edr_files: List[str], out_file: str):
    cmd = ["gmx", "eneconv", "-f"] + edr_files + ["-o", out_file]
    subprocess.run(cmd, check=True)


def extract_potential(edr_file: str, out_file: str):
    cmd = ["gmx", "energy", "-f", edr_file, "-o", out_file]
    subprocess.run(cmd, input="Potential\n", text=True, check=True)


def main():
    parser = _argparse.ArgumentParser(description="YAGTRAJ - GROMACS MD Analysis")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run interactive CLI"
    )
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")
    parser.add_argument("--input", type=str, help="Input directory for T-REMD analysis")
    parser.add_argument("--out", type=str, help="Output directory for T-REMD analysis")
    parser.add_argument("--deffnm", default="remd", help="Base name for trajectory/log files (T-REMD)")
    parser.add_argument("--demux-script", default="demux.pl", help="Demux script (default: demux.pl) (T-REMD)")
    args = parser.parse_args()
    cli = YagtrajShell("gmx")
    if args.file:
        try:
            with open(args.file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        print(f"YAGTRAJ> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"[!] File '{args.file}' not found.")
            _sys.exit(1)
    else:
        cli.cmdloop()


# Patch the entrypoint
if __name__ == "__main__":
    main()
