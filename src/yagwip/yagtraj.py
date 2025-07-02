"""
yagtraj.py: Trajectory analysis tool for Gromacs MD simulations

Portions copyright (c) 2025 the Authors.
Authors: Nathan Levinzon, Olivier Mailhot
Contributors:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import cmd
import sys
import os
from .utils import *
from importlib.resources import files
import importlib.metadata
import random

__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YAGTRAJ_shell(cmd.Cmd):
    intro = f"Welcome to YAGTRAJ v{__version__}. Type help to list commands."
    prompt = "YAGTRAJ> "

    def __init__(self, gmx_path):
        super().__init__()
        self.debug = False  # Toggle debug mode
        self.gmx_path = gmx_path  # Path to GROMACS executable (e.g., "gmx")
        self.logger = setup_logger(debug_mode=self.debug)  # Initialize logging
        self.current_tpr = None  # Current TPR file
        self.current_traj = None  # Current trajectory file
        self.print_banner()  # Prints intro banner to command line

        # Validate GROMACS installation
        try:
            validate_gromacs_installation(gmx_path)
        except RuntimeError as e:
            print(f"[!] GROMACS Validation Error: {e}")
            print(
                "[!] YAGWIP cannot start without GROMACS. Please install GROMACS and try again."
            )
            sys.exit(1)

    def _log(self, msg):
        """Log message using logger or print if no logger available."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def default(self, line):
        """Throws error when command is not recognized"""
        self._log(f"[!] Unknown command: {line}")

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
        self.logger = setup_logger(debug_mode=self.debug)

        self._log(f"[DEBUG] Debug mode is now {'ON' if self.debug else 'OFF'}")

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
            self._log(f"[!] Could not load banner: {e}")

    def do_load(self, arg):
        """
        Load trajectory files for analysis.
        Usage: load <tpr_file> [traj_file]
        Example: load production.tpr production.xtc
        """
        args = arg.strip().split()
        if len(args) < 1:
            self._log("[!] Usage: load <tpr_file> [traj_file]")
            return

        tpr_file = args[0]
        if not os.path.exists(tpr_file):
            self._log(f"[!] TPR file '{tpr_file}' not found.")
            return

        self.current_tpr = tpr_file
        self._log(f"[#] Loaded TPR file: {tpr_file}")

        if len(args) > 1:
            traj_file = args[1]
            if not os.path.exists(traj_file):
                self._log(f"[!] Trajectory file '{traj_file}' not found.")
                return
            self.current_traj = traj_file
            self._log(f"[#] Loaded trajectory file: {traj_file}")

    def _require_files(self):
        """Check if required files are loaded."""
        if not self.current_tpr:
            self._log("[!] No TPR file loaded. Use 'load <tpr_file>' first.")
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

        self._log(f"[RMSD] Calculating RMSD for protein backbone...")

        # Use run_gromacs_command from utils
        success = run_gromacs_command(
            command=command,
            pipe_input="4\n4\n",  # Select backbone for both reference and fit
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(f"[RMSD] RMSD calculation completed. Output: {output_file}")
        else:
            self._log("[!] RMSD calculation failed.")

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

        self._log(f"[RGYR] Calculating radius of gyration...")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n",  # Select protein
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(
                f"[RGYR] Radius of gyration calculation completed. Output: {output_file}"
            )
        else:
            self._log("[!] Radius of gyration calculation failed.")

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

        self._log(f"[SASA] Calculating solvent accessible surface area...")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n",  # Select protein
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(f"[SASA] SASA calculation completed. Output: {output_file}")
        else:
            self._log("[!] SASA calculation failed.")

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
        command += f" -num {output_file}"

        self._log(f"[HBOND] Calculating hydrogen bonds...")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n1\n",  # Select protein for both donor and acceptor
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(
                f"[HBOND] Hydrogen bond calculation completed. Output: {output_file}"
            )
        else:
            self._log("[!] Hydrogen bond calculation failed.")

    def do_distance(self, arg):
        """
        Calculate distance between two groups.
        Usage: distance <group1> <group2> [output_file]
        Example: distance 1 2 distance.xvg
        """
        if not self._require_files():
            return

        args = arg.strip().split()
        if len(args) < 2:
            self._log("[!] Usage: distance <group1> <group2> [output_file]")
            return

        group1, group2 = args[0], args[1]
        output_file = args[2] if len(args) > 2 else "distance.xvg"

        command = f"{self.gmx_path} distance -s {self.current_tpr}"
        if self.current_traj:
            command += f" -f {self.current_traj}"
        command += f" -o {output_file}"

        self._log(
            f"[DISTANCE] Calculating distance between groups {group1} and {group2}..."
        )

        success = run_gromacs_command(
            command=command,
            pipe_input=f"{group1}\n{group2}\n",
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(
                f"[DISTANCE] Distance calculation completed. Output: {output_file}"
            )
        else:
            self._log("[!] Distance calculation failed.")

    def do_energy(self, arg):
        """
        Extract energy terms from energy file.
        Usage: energy <edr_file> [output_file]
        """
        args = arg.strip().split()
        if len(args) < 1:
            self._log("[!] Usage: energy <edr_file> [output_file]")
            return

        edr_file = args[0]
        if not os.path.exists(edr_file):
            self._log(f"[!] Energy file '{edr_file}' not found.")
            return

        output_file = args[1] if len(args) > 1 else "energy.xvg"

        command = f"{self.gmx_path} energy -f {edr_file} -o {output_file}"

        self._log(f"[ENERGY] Extracting energy terms from {edr_file}...")

        success = run_gromacs_command(
            command=command,
            pipe_input="10\n11\n0\n",  # Select potential and kinetic energy, then quit
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(f"[ENERGY] Energy extraction completed. Output: {output_file}")
        else:
            self._log("[!] Energy extraction failed.")

    def do_trjconv(self, arg):
        """
        Convert trajectory format or extract frames.
        Usage: trjconv <input_traj> <output_traj> [options]
        Example: trjconv traj.xtc traj.pdb -dump 1000
        """
        args = arg.strip().split()
        if len(args) < 2:
            self._log("[!] Usage: trjconv <input_traj> <output_traj> [options]")
            return

        input_traj = args[0]
        output_traj = args[1]

        if not os.path.exists(input_traj):
            self._log(f"[!] Input trajectory '{input_traj}' not found.")
            return

        if not self.current_tpr:
            self._log("[!] No TPR file loaded. Use 'load <tpr_file>' first.")
            return

        # Build command with additional options
        command = f"{self.gmx_path} trjconv -s {self.current_tpr} -f {input_traj} -o {output_traj}"

        # Add any additional options
        if len(args) > 2:
            command += " " + " ".join(args[2:])

        self._log(f"[TRJCONV] Converting trajectory...")

        success = run_gromacs_command(
            command=command,
            pipe_input="1\n",  # Select protein
            debug=self.debug,
            logger=self.logger,
        )

        if success:
            self._log(
                f"[TRJCONV] Trajectory conversion completed. Output: {output_traj}"
            )
        else:
            self._log("[!] Trajectory conversion failed.")

    def do_tremd_demux(self, arg):
        """
        Demultiplex TREMD trajectories and logs.
        Usage: tremd demux [base_name]
        Example: tremd demux remd
        """
        base_name = arg.strip() if arg.strip() else "remd"

        # Count replica directories
        replica_dirs = []
        for item in os.listdir("."):
            if os.path.isdir(item) and item.isdigit():
                replica_dirs.append(int(item))

        if not replica_dirs:
            self._log(
                "[!] No replica directories found (directories named with digits only)"
            )
            return

        replica_dirs.sort()
        num_replicas = len(replica_dirs)
        self._log(f"[TREMD] Found {num_replicas} TREMD directories: {replica_dirs}")

        # Create analysis directory
        analysis_dir = "remd_analysis_results"
        if not self.debug:
            os.makedirs(analysis_dir, exist_ok=True)

        # Create temporary directory for logs
        log_tmp = "remd_logs"
        if not self.debug:
            os.makedirs(log_tmp, exist_ok=True)

        # Copy and concatenate logs
        self._log("[TREMD] Processing replica logs...")
        for replica in replica_dirs:
            log_file = f"{replica}/{base_name}.log"
            if os.path.exists(log_file):
                if not self.debug:
                    import shutil

                    shutil.copy(log_file, f"{log_tmp}/remd_{replica}.log")
            else:
                self._log(f"[!] Warning: {log_file} not found")

        # Concatenate logs
        if not self.debug:
            with open(f"{log_tmp}/REMD.log", "w") as outfile:
                for replica in replica_dirs:
                    log_file = f"{log_tmp}/remd_{replica}.log"
                    if os.path.exists(log_file):
                        with open(log_file, "r") as infile:
                            outfile.write(infile.read())

        # Run demux.pl (if available) or use gmx demux
        self._log("[TREMD] Generating replica index...")

        # Try to use demux.pl first, fallback to gmx demux
        demux_command = f"demux.pl {log_tmp}/REMD.log"
        success = run_gromacs_command(
            command=demux_command, debug=self.debug, logger=self.logger
        )

        if not success:
            # Fallback to gmx demux
            self._log("[TREMD] demux.pl not found, trying gmx demux...")
            demux_command = f"{self.gmx_path} demux {log_tmp}/REMD.log"
            success = run_gromacs_command(
                command=demux_command, debug=self.debug, logger=self.logger
            )

        if success and not self.debug:
            # Move generated files to analysis directory
            for file in ["replica_index.xvg", "replica_temp.xvg"]:
                if os.path.exists(file):
                    import shutil

                    shutil.move(file, f"{analysis_dir}/{file}")

        # Demultiplex trajectories
        self._log("[TREMD] Demultiplexing trajectories...")

        # Build list of trajectory files
        traj_files = []
        for replica in replica_dirs:
            traj_file = f"{replica}/{base_name}.xtc"
            if os.path.exists(traj_file):
                traj_files.append(traj_file)
            else:
                self._log(f"[!] Warning: {traj_file} not found")

        if traj_files:
            # Use gmx trjcat with demux
            trjcat_command = f"{self.gmx_path} trjcat -f {' '.join(traj_files)} -demux {analysis_dir}/replica_index.xvg"

            success = run_gromacs_command(
                command=trjcat_command, debug=self.debug, logger=self.logger
            )

            if success and not self.debug:
                # Move demuxed trajectories to analysis directories
                for replica in replica_dirs:
                    trajout_file = f"{replica}_trajout.xtc"
                    if os.path.exists(trajout_file):
                        replica_dir = f"{analysis_dir}/replica_{replica}"
                        os.makedirs(replica_dir, exist_ok=True)
                        import shutil

                        shutil.move(trajout_file, f"{replica_dir}/{trajout_file}")

                        # Copy corresponding TPR file
                        tpr_file = f"{replica}/{base_name}.tpr"
                        if os.path.exists(tpr_file):
                            shutil.copy(
                                tpr_file, f"{replica_dir}/demuxed_{replica}.tpr"
                            )

        self._log(f"[TREMD] Demultiplexing completed. Results in {analysis_dir}/")

    def do_tremd_rmsd(self, arg):
        """
        Calculate RMSD for all TREMD replicas.
        Usage: tremd rmsd [base_name]
        Example: tremd rmsd remd
        """
        base_name = arg.strip() if arg.strip() else "remd"
        analysis_dir = "remd_analysis_results"

        if not os.path.exists(analysis_dir):
            self._log("[!] Analysis directory not found. Run 'tremd demux' first.")
            return

        # Find replica directories
        replica_dirs = []
        for item in os.listdir(analysis_dir):
            if item.startswith("replica_") and os.path.isdir(
                os.path.join(analysis_dir, item)
            ):
                replica_num = item.split("_")[1]
                if replica_num.isdigit():
                    replica_dirs.append(int(replica_num))

        if not replica_dirs:
            self._log(
                "[!] No replica analysis directories found. Run 'tremd demux' first."
            )
            return

        replica_dirs.sort()
        self._log(f"[TREMD] Calculating RMSD for {len(replica_dirs)} replicas...")

        for replica in replica_dirs:
            replica_dir = f"{analysis_dir}/replica_{replica}"
            tpr_file = f"{replica_dir}/demuxed_{replica}.tpr"
            traj_file = f"{replica_dir}/{replica}_trajout.xtc"
            output_file = f"{replica_dir}/rmsd.xvg"

            if not os.path.exists(tpr_file) or not os.path.exists(traj_file):
                self._log(f"[!] Warning: Missing files for replica {replica}")
                continue

            command = f"{self.gmx_path} rms -s {tpr_file} -f {traj_file} -o {output_file} -res"

            self._log(f"[TREMD] Processing replica {replica} RMSD...")

            success = run_gromacs_command(
                command=command,
                pipe_input="4\n4\n",  # Select backbone for both reference and fit
                debug=self.debug,
                logger=self.logger,
            )

            if success:
                self._log(f"[TREMD] Replica {replica} RMSD completed: {output_file}")
            else:
                self._log(f"[!] Replica {replica} RMSD failed")

    def do_tremd_rmsf(self, arg):
        """
        Calculate RMSF for all TREMD replicas.
        Usage: tremd rmsf [base_name]
        Example: tremd rmsf remd
        """
        base_name = arg.strip() if arg.strip() else "remd"
        analysis_dir = "remd_analysis_results"

        if not os.path.exists(analysis_dir):
            self._log("[!] Analysis directory not found. Run 'tremd demux' first.")
            return

        # Find replica directories
        replica_dirs = []
        for item in os.listdir(analysis_dir):
            if item.startswith("replica_") and os.path.isdir(
                os.path.join(analysis_dir, item)
            ):
                replica_num = item.split("_")[1]
                if replica_num.isdigit():
                    replica_dirs.append(int(replica_num))

        if not replica_dirs:
            self._log(
                "[!] No replica analysis directories found. Run 'tremd demux' first."
            )
            return

        replica_dirs.sort()
        self._log(f"[TREMD] Calculating RMSF for {len(replica_dirs)} replicas...")

        for replica in replica_dirs:
            replica_dir = f"{analysis_dir}/replica_{replica}"
            tpr_file = f"{replica_dir}/demuxed_{replica}.tpr"
            traj_file = f"{replica_dir}/{replica}_trajout.xtc"
            output_file = f"{replica_dir}/rmsf.xvg"

            if not os.path.exists(tpr_file) or not os.path.exists(traj_file):
                self._log(f"[!] Warning: Missing files for replica {replica}")
                continue

            command = f"{self.gmx_path} rmsf -s {tpr_file} -f {traj_file} -o {output_file} -res"

            self._log(f"[TREMD] Processing replica {replica} RMSF...")

            success = run_gromacs_command(
                command=command,
                pipe_input="4\n",  # Select backbone
                debug=self.debug,
                logger=self.logger,
            )

            if success:
                self._log(f"[TREMD] Replica {replica} RMSF completed: {output_file}")
            else:
                self._log(f"[!] Replica {replica} RMSF failed")

    def do_tremd_pca(self, arg):
        """
        Perform PCA analysis for all TREMD replicas.
        Usage: tremd pca [base_name]
        Example: tremd pca remd
        """
        base_name = arg.strip() if arg.strip() else "remd"
        analysis_dir = "remd_analysis_results"

        if not os.path.exists(analysis_dir):
            self._log("[!] Analysis directory not found. Run 'tremd demux' first.")
            return

        # Find replica directories
        replica_dirs = []
        for item in os.listdir(analysis_dir):
            if item.startswith("replica_") and os.path.isdir(
                os.path.join(analysis_dir, item)
            ):
                replica_num = item.split("_")[1]
                if replica_num.isdigit():
                    replica_dirs.append(int(replica_num))

        if not replica_dirs:
            self._log(
                "[!] No replica analysis directories found. Run 'tremd demux' first."
            )
            return

        replica_dirs.sort()
        self._log(f"[TREMD] Performing PCA for {len(replica_dirs)} replicas...")

        for replica in replica_dirs:
            replica_dir = f"{analysis_dir}/replica_{replica}"
            tpr_file = f"{replica_dir}/demuxed_{replica}.tpr"
            traj_file = f"{replica_dir}/{replica}_trajout.xtc"

            if not os.path.exists(tpr_file) or not os.path.exists(traj_file):
                self._log(f"[!] Warning: Missing files for replica {replica}")
                continue

            self._log(f"[TREMD] Processing replica {replica} PCA...")

            # Step 1: Center trajectory
            traj_centered = f"{replica_dir}/traj_centered.xtc"
            command1 = f"{self.gmx_path} trjconv -s {tpr_file} -f {traj_file} -o {traj_centered} -center -pbc mol"

            success1 = run_gromacs_command(
                command=command1,
                pipe_input="0\n0\n",  # Select system for centering
                debug=self.debug,
                logger=self.logger,
            )

            if not success1:
                self._log(f"[!] Replica {replica} PCA step 1 (centering) failed")
                continue

            # Step 2: Fit trajectory
            traj_fitted = f"{replica_dir}/traj_centered_rot_trans.xtc"
            command2 = f"{self.gmx_path} trjconv -s {tpr_file} -f {traj_centered} -o {traj_fitted} -ur compact -fit rot+trans"

            success2 = run_gromacs_command(
                command=command2,
                pipe_input="0\n0\n",  # Select system for fitting
                debug=self.debug,
                logger=self.logger,
            )

            if not success2:
                self._log(f"[!] Replica {replica} PCA step 2 (fitting) failed")
                continue

            # Step 3: Calculate covariance matrix
            eigenval_file = f"{replica_dir}/eigenval.xvg"
            eigenvec_file = f"{replica_dir}/eigenvec.trr"
            command3 = f"{self.gmx_path} covar -s {tpr_file} -f {traj_fitted} -o {eigenval_file} -v {eigenvec_file}"

            success3 = run_gromacs_command(
                command=command3,
                pipe_input="4\n4\n",  # Select backbone for covariance
                debug=self.debug,
                logger=self.logger,
            )

            if not success3:
                self._log(f"[!] Replica {replica} PCA step 3 (covariance) failed")
                continue

            # Step 4: Project trajectory
            proj_file = f"{replica_dir}/proj.xvg"
            command4 = f"{self.gmx_path} anaeig -v {eigenvec_file} -s {tpr_file} -f {traj_fitted} -proj {proj_file}"

            success4 = run_gromacs_command(
                command=command4,
                pipe_input="4\n4\n",  # Select backbone for projection
                debug=self.debug,
                logger=self.logger,
            )

            if success4:
                self._log(f"[TREMD] Replica {replica} PCA completed: {proj_file}")
            else:
                self._log(f"[!] Replica {replica} PCA step 4 (projection) failed")

    def do_tremd_temp(self, arg):
        """
        Extract temperature data for all TREMD replicas.
        Usage: tremd temp [base_name]
        Example: tremd temp remd
        """
        base_name = arg.strip() if arg.strip() else "remd"
        analysis_dir = "remd_analysis_results"

        if not os.path.exists(analysis_dir):
            self._log("[!] Analysis directory not found. Run 'tremd demux' first.")
            return

        # Find replica directories
        replica_dirs = []
        for item in os.listdir(analysis_dir):
            if item.startswith("replica_") and os.path.isdir(
                os.path.join(analysis_dir, item)
            ):
                replica_num = item.split("_")[1]
                if replica_num.isdigit():
                    replica_dirs.append(int(replica_num))

        if not replica_dirs:
            self._log(
                "[!] No replica analysis directories found. Run 'tremd demux' first."
            )
            return

        replica_dirs.sort()
        self._log(f"[TREMD] Extracting temperature for {len(replica_dirs)} replicas...")

        for replica in replica_dirs:
            replica_dir = f"{analysis_dir}/replica_{replica}"
            original_edr = f"{replica}/{base_name}.edr"
            output_edr = f"{replica_dir}/ener.edr"
            output_temp = f"{replica_dir}/temp.xvg"

            if not os.path.exists(original_edr):
                self._log(
                    f"[!] Warning: {original_edr} not found for replica {replica}"
                )
                continue

            # Copy energy file to analysis directory
            if not self.debug:
                import shutil

                shutil.copy(original_edr, output_edr)

            # Extract temperature
            command = f"{self.gmx_path} energy -f {output_edr} -o {output_temp}"

            self._log(f"[TREMD] Processing replica {replica} temperature...")

            success = run_gromacs_command(
                command=command,
                pipe_input="Temperature\n",  # Select temperature
                debug=self.debug,
                logger=self.logger,
            )

            if success:
                self._log(
                    f"[TREMD] Replica {replica} temperature completed: {output_temp}"
                )
            else:
                self._log(f"[!] Replica {replica} temperature extraction failed")

    def do_tremd_energy(self, arg):
        """
        Combine and analyze energy files from all TREMD replicas.
        Usage: tremd energy [base_name]
        Example: tremd energy remd
        """
        base_name = arg.strip() if arg.strip() else "remd"
        analysis_dir = "remd_analysis_results"

        if not os.path.exists(analysis_dir):
            self._log("[!] Analysis directory not found. Run 'tremd demux' first.")
            return

        # Find all energy files
        energy_files = []
        for item in os.listdir("."):
            if os.path.isdir(item) and item.isdigit():
                edr_file = f"{item}/{base_name}.edr"
                if os.path.exists(edr_file):
                    energy_files.append(edr_file)

        if not energy_files:
            self._log("[!] No energy files found in replica directories")
            return

        energy_files.sort()
        self._log(f"[TREMD] Found {len(energy_files)} energy files")

        # Combine energy files
        combined_edr = f"{analysis_dir}/combined.edr"
        command1 = (
            f"{self.gmx_path} eneconv -f {' '.join(energy_files)} -o {combined_edr}"
        )

        self._log("[TREMD] Combining energy files...")

        success1 = run_gromacs_command(
            command=command1, debug=self.debug, logger=self.logger
        )

        if not success1:
            self._log("[!] Energy file combination failed")
            return

        # Extract potential energy
        output_energy = f"{analysis_dir}/kbT_scalar.xvg"
        command2 = f"{self.gmx_path} energy -f {combined_edr} -o {output_energy}"

        self._log("[TREMD] Extracting potential energy...")

        success2 = run_gromacs_command(
            command=command2,
            pipe_input="Potential\n",  # Select potential energy
            debug=self.debug,
            logger=self.logger,
        )

        if success2:
            self._log(f"[TREMD] Energy analysis completed: {output_energy}")
        else:
            self._log("[!] Energy extraction failed")

    def do_info(self, arg):
        """Show information about loaded files."""
        self._log("[INFO] Current file status:")
        self._log(f"  TPR file: {self.current_tpr or 'None'}")
        self._log(f"  Trajectory file: {self.current_traj or 'None'}")
        self._log(f"  GROMACS path: {self.gmx_path}")
        self._log(f"  Debug mode: {'ON' if self.debug else 'OFF'}")

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
                print(f"\nYAGTRAJ Reminds You...\n{random.choice(quotes)}\n")
        except Exception as e:
            self._log(f"[!] Unable to load quotes: {e}")

    def do_quit(self, _):
        """
        Quit the CLI.
        Usage: "quit"
        """
        self.print_random_quote()
        self._log(f"Copyright (c) 2025 {__author__} \nQuitting YAGTRAJ.")
        return True


def main():
    parser = argparse.ArgumentParser(description="YAGTRAJ - GROMACS MD Analysis")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run interactive CLI"
    )
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")

    args = parser.parse_args()
    cli = YAGTRAJ_shell("gmx")

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
