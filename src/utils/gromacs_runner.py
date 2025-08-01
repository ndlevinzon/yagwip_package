"""
gromacs_runner.py: GROMACS and ligand building utilities for YAGWIP.
"""

# === Standard Library Imports ===
import os
import shutil
from typing import List
from importlib.resources import files

# === Local Imports ===
from yagwip.base import YagwipBase
from utils.log_utils import auto_monitor

# Constants for GROMACS command inputs
PIPE_INPUTS = {"pdb2gmx": "12\n", "genion_prot": "13\n", "genion_complex": "15\n", "genion_lig": "4\n"}


class Builder(YagwipBase):
    """Handles GROMACS system building steps."""

    def __init__(self, gmx_path, debug=False, logger=None):
        """Initialize Builder."""
        super().__init__(gmx_path=gmx_path, debug=debug, logger=logger)

    @auto_monitor
    def _resolve_basename(self, basename):
        """Resolve the basename for file operations."""
        if not basename and not self.debug:
            self._log_error("No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return None
        return basename if basename else "PLACEHOLDER"

    @auto_monitor
    def run_pdb2gmx(self, basename, custom_command=None):
        """Run pdb2gmx to generate topology and coordinates."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        cmd = custom_command or (
            f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water tip3p -ignh"
        )
        if self.debug:
            print(f"[DEBUG] Command: {cmd}")
            return
        self._log(f"Running pdb2gmx for {base}.pdb...")
        self._execute_command(
            cmd, f"pdb2gmx for {base}", pipe_input=PIPE_INPUTS["pdb2gmx"]
        )

    @auto_monitor
    def run_solvate(self, basename, custom_command=None):
        """Run solvate to add solvent to the system."""
        base = self._resolve_basename(basename)
        if base is None:
            return
        default_box = " -c -d 1.0 -bt cubic"
        default_water = "spc216.gro"

        # Use the topol.top file in the current working directory
        topol_path = "topol.top"
        if not os.path.exists(topol_path):
            self._log_error(f"topol.top not found in current directory: {os.getcwd()}")
            return

        default_cmds = [
            f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro {default_box}",
            f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs {default_water} -o {base}.solv.gro -p {topol_path}",
        ]
        if custom_command:
            self._log_info("Using custom solvate command")
            self._execute_command(custom_command, "custom solvate")
        else:
            for i, cmd in enumerate(default_cmds):
                self._execute_command(cmd, f"solvate step {i+1}")

    @auto_monitor
    def run_genions(self, basename, custom_command=None):
        """Run genion to add ions to the system. If lambda directories are present, copy and
        patch ions_fep.mdp in each lambda dir with correct lambda index and run genions in each."""
        # Non-FEP or no lambda dirs: original logic
        base = self._resolve_basename(basename)
        if base is None:
            return
        else:
            mdp_file = files("templates").joinpath("ions.mdp")
        input_gro = f"{base}.solv.gro"
        output_gro = f"{base}.solv.ions.gro"
        tpr_out = "ions.tpr"
        ion_options = "-pname NA -nname CL -conc 0.150 -neutral"
        grompp_opts = ""
        ion_pipe_input = (
            PIPE_INPUTS["genion_lig"]
            if base.startswith("ligand") or base.startswith("hybrid")
            else PIPE_INPUTS["genion_prot"]
            if base.endswith("protein")
            else PIPE_INPUTS["genion_complex"]
        )
        default_cmds = [
            f"{self.gmx_path} grompp -f {mdp_file} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts} -maxwarn 50",
            f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}",
        ]
        self._log_info(f"Running genion for {base}")
        if custom_command:
            self._log_info("Using custom genion command")
            self._execute_command(custom_command, "custom genion")
        else:
            for i, cmd in enumerate(default_cmds):
                pipe_input = ion_pipe_input if i == 1 else None
                self._execute_command(cmd, f"genion step {i+1}", pipe_input=pipe_input)


class GromacsCommands(YagwipBase):
    def __init__(self, gmx_path, debug=False, logger=None):
        super().__init__(gmx_path=gmx_path, debug=debug, logger=logger)

    def run_em(self, basename, arg=""):
        self._run_stage(
            basename, arg, default_mdp="em.mdp", suffix=".solv.ions", tprname="em"
        )

    def run_nvt(self, basename, arg=""):
        self._run_stage(
            basename, arg, default_mdp="nvt.mdp", suffix=".em", tprname="nvt"
        )

    def run_npt(self, basename, arg=""):
        self._run_stage(
            basename, arg, default_mdp="npt.mdp", suffix=".nvt", tprname="npt"
        )

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

        self._execute_command(grompp_cmd, "grompp for production")
        self._execute_command(mdrun_cmd, "mdrun for production")

    def _run_stage(self, basename, arg, default_mdp, suffix, tprname):
        base = basename if basename else "PLACEHOLDER"
        self._log_info(f"Running stage for {base} using {default_mdp}")

        parts = arg.strip().split(maxsplit=3)
        mdpfile = (
            parts[0]
            if len(parts) > 0
            else str(files("yagwip.templates").joinpath(default_mdp))
        )
        suffix = parts[1] if len(parts) > 1 else suffix
        tprname = parts[2] if len(parts) > 2 else tprname
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        input_gro = f"{base}{suffix}.gro"
        tpr_file = f"{tprname}.tpr"

        grompp_cmd = f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_file}"
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

        self._execute_command(grompp_cmd, f"grompp for {tprname}")
        self._execute_command(mdrun_cmd, f"mdrun for {tprname}")

    def run_autoimage(self, basename, arg=""):
        """
        Run autoimage workflow to process trajectory files.

        This method executes a series of trjconv commands to:
        1. Apply periodic boundary conditions (whole molecules)
        2. Center the system
        3. Create a PDB file with proper imaging

        Args:
            basename: Base name for the files (defaults to 'production')
            arg: Optional arguments (currently unused, for future expansion)
        """
        # Use production as default basename if none provided
        base = basename if basename else "production"

        self._log_info(f"Running autoimage workflow for {base}")

        # Step 1: Apply periodic boundary conditions (whole molecules)
        cmd1 = f"{self.gmx_path} trjconv -s {base}.tpr -f {base}.xtc -o {base}.pbc1.xtc -pbc whole -ur compact"
        self._execute_command(cmd1, "trjconv step 1: apply PBC whole")

        # Step 2: Center the system
        cmd2 = f"{self.gmx_path} trjconv -s {base}.tpr -f {base}.pbc1.xtc -o {base}.noPBC.xtc -center -n"
        self._execute_command(cmd2, "trjconv step 2: center system")

        # Step 3: Create PDB file with proper imaging
        cmd3 = f"{self.gmx_path} trjconv -s {base}.tpr -f {base}.noPBC.xtc -o {base}.pdb -pbc mol -ur compact"
        self._execute_command(cmd3, "trjconv step 3: create PDB with proper imaging")

        self._log_success(f"Autoimage workflow completed for {base}")

    def run_demux(self, input_dir: str, arg: str = ""):
        """
        Run demultiplexing workflow for replica exchange simulations.

        This method performs the complete demultiplexing process:
        1. Detects replica directories
        2. Aggregates log files
        3. Runs demux script to generate index files
        4. Demultiplexes trajectories

        Args:
            input_dir: Directory containing replica subdirectories
            arg: Optional arguments (currently unused, for future expansion)
        """
        self._log_info(f"Running demux workflow for directory: {input_dir}")

        # Step 1: Detect replica directories
        replicas = self._detect_replicas(input_dir)
        if not replicas:
            self._log_error(f"No replica directories found in {input_dir}")
            return

        self._log_info(f"Found {len(replicas)} replica directories: {replicas}")

        # Step 2: Create temporary directory for logs
        log_tmp = os.path.join(input_dir, "log_tmp")
        os.makedirs(log_tmp, exist_ok=True)

        # Step 3: Aggregate log files
        self._aggregate_logs(replicas, input_dir, log_tmp)

        # Step 4: Run demux script
        log_file = os.path.join(log_tmp, "REMD.log")
        if not os.path.exists(log_file):
            self._log_error(f"REMD log file not found: {log_file}")
            return

        self._run_demux_script(log_file, log_tmp)

        # Step 5: Demultiplex trajectories
        self._demux_trajectories(input_dir, replicas, log_tmp)

        self._log_success(f"Demux workflow completed for {len(replicas)} replicas")

    def _detect_replicas(self, input_dir: str) -> List[str]:
        """Detect replica directories named as integers."""
        if not os.path.exists(input_dir):
            self._log_error(f"Input directory does not exist: {input_dir}")
            return []

        replicas = []
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if item.isdigit() and os.path.isdir(item_path):
                replicas.append(item)

        return sorted(replicas, key=int)

    def _aggregate_logs(self, replicas: List[str], input_dir: str, log_tmp: str):
        """Aggregate log files from all replicas."""
        self._log_info("Aggregating log files from replicas...")

        for replica in replicas:
            src = os.path.join(input_dir, replica, "md.log")
            dst = os.path.join(log_tmp, f"remd_{replica}.log")

            if os.path.exists(src):
                shutil.copy2(src, dst)
                self._log_debug(f"Copied log from replica {replica}")
            else:
                self._log_warning(f"Log file not found for replica {replica}: {src}")

        # Create combined REMD.log file
        combined_log = os.path.join(log_tmp, "REMD.log")
        with open(combined_log, "w") as outfile:
            for replica in replicas:
                log_path = os.path.join(log_tmp, f"remd_{replica}.log")
                if os.path.exists(log_path):
                    with open(log_path, "r") as infile:
                        outfile.write(infile.read())

        self._log_info(f"Created combined log file: {combined_log}")

    def _run_demux_script(self, log_file: str, out_dir: str, demux_script: str = "demux.pl"):
        """Run the demux script to generate index files."""
        self._log_info(f"Running demux script: {demux_script}")

        try:
            # Check if demux script exists
            if not shutil.which(demux_script):
                self._log_error(f"Demux script not found in PATH: {demux_script}")
                return

            # Run demux script
            cmd = [demux_script, log_file]
            self._execute_command(" ".join(cmd), f"demux script execution")

            # Move generated files to output directory
            for fname in ["replica_index.xvg", "replica_temp.xvg"]:
                if os.path.exists(fname):
                    shutil.move(fname, out_dir)
                    self._log_info(f"Moved {fname} to {out_dir}")
                else:
                    self._log_warning(f"Expected file not generated: {fname}")

        except Exception as e:
            self._log_error(f"Failed to run demux script: {e}")

    def _demux_trajectories(self, input_dir: str, replicas: List[str], log_tmp: str):
        """Demultiplex trajectories using the generated index file."""
        self._log_info("Demultiplexing trajectories...")

        # Find trajectory files
        xtc_files = []
        for replica in replicas:
            xtc_path = os.path.join(input_dir, replica, "md.xtc")
            if os.path.exists(xtc_path):
                xtc_files.append(xtc_path)
            else:
                self._log_warning(f"Trajectory file not found for replica {replica}: {xtc_path}")

        if not xtc_files:
            self._log_error("No trajectory files found for demultiplexing")
            return

        # Check for index file
        index_file = os.path.join(log_tmp, "replica_index.xvg")
        if not os.path.exists(index_file):
            self._log_error(f"Index file not found: {index_file}")
            return

        # Run trjcat with demux
        try:
            cmd = [self.gmx_path, "trjcat", "-f"] + xtc_files + ["-demux", index_file]
            self._execute_command(" ".join(cmd), "trajectory demultiplexing")
            self._log_success("Trajectory demultiplexing completed")
        except Exception as e:
            self._log_error(f"Failed to demultiplex trajectories: {e}")
