"""
YAGWIP: Yet Another GROMACS Wrapper In Python

A comprehensive Python-native application and library that automates the setup
and execution of GROMACS molecular dynamics (MD) simulations, including support
for both standard and advanced simulation types like Temperature Replica Exchange
Molecular Dynamics (TREMD) and Free Energy Perturbation (FEP).

This module provides the main interactive command-line interface (CLI) for YAGWIP,
offering a user-friendly shell environment for molecular simulation workflows.

Key Features:
- Interactive CLI with tab completion and command history
- Support for protein-only, ligand-only, and protein-ligand systems
- Automated FEP workflow with hybrid topology generation
- TREMD temperature ladder calculation and setup
- Batch processing capabilities for multiple PDB files
- Custom command override system
- Comprehensive logging and debugging tools

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

# === Standard Library Imports ===
import os
import re
import io
import cmd
import sys
import string
import subprocess
import random
import argparse
import shlex
import shutil
import importlib.metadata
import multiprocessing as mp
from pathlib import Path
from importlib.resources import files
from typing import Optional, List, Dict, Tuple

# === Third-Party Imports ===
import pandas as pd

# === Local Imports ===
from utils.gromacs_runner import Builder, GromacsCommands
from yagwip.ligand_builder import LigandPipeline
from yagwip.base import YagwipBase
from yagwip.config import validate_gromacs_installation, detect_gromacs_executable
from utils.slurm_utils import SlurmWriter
from utils.pipeline_utils import Editor
from utils.log_utils import setup_logger
from utils.batch_processor import ParallelBatchProcessor

# === Metadata ===
__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YagwipShell(cmd.Cmd, YagwipBase):
    """
    Interactive shell for YAGWIP: Yet Another GROMACS Wrapper In Python.

    This class provides a comprehensive command-line interface for molecular
    simulation workflows, extending Python's cmd.Cmd with GROMACS-specific
    functionality and YAGWIP's base features.

    The shell supports three main workflow types:
    1. Protein-only simulations
    2. Protein-ligand complex simulations
    3. Free Energy Perturbation (FEP) workflows
    4. Temperature Replica Exchange MD (TREMD)

    Key Features:
    - Interactive command execution with tab completion
    - Automatic file detection and workflow selection
    - Custom command override system
    - Batch processing capabilities
    - Comprehensive error handling and logging
    - Debug mode with detailed system monitoring

    Attributes:
        current_pdb_path (Optional[str]): Full path to the loaded PDB file
        ligand_pdb_path (Optional[str]): Full path to the ligand PDB file, if any
        basename (Optional[str]): Base PDB filename (without extension)
        user_itp_paths (List[str]): Stores user input paths for do_source
        editor (Editor): File editor instance for topology manipulation
        ligand_pipeline (LigandPipeline): Ligand processing pipeline
        gmx (GromacsCommands): Simulation execution handler
        builder (Builder): System building handler
        custom_cmds (Dict[str, str]): Custom command overrides
        ligand_counter (int): Counter for FEP-style ligand naming
        current_ligand_name (Optional[str]): Current ligand name being processed
    """

    # Intro message and prompt for the interactive CLI
    intro = f"Welcome to YAGWIP v{__version__}. Type help to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path: str = None) -> None:
        """
        Initialize the YAGWIP shell with GROMACS path.

        Sets up the interactive shell environment, initializes all necessary
        components (Builder, Sim, LigandPipeline, Editor), validates GROMACS
        installation, and prepares the shell for command execution.

        Args:
            gmx_path: Path to GROMACS executable (if None, will auto-detect)

        Raises:
            RuntimeError: If GROMACS installation validation fails
            SystemExit: If critical initialization fails
        """
        # Auto-detect GROMACS executable if not provided
        if gmx_path is None:
            try:
                gmx_path = detect_gromacs_executable()
                self._log_info(f"Auto-detected GROMACS executable: {gmx_path}")
            except RuntimeError as e:
                self._log_error(f"GROMACS Detection Error: {e}")
                sys.exit(1)

        # Initialize cmd.Cmd first (no parameters)
        cmd.Cmd.__init__(self)
        # Initialize YagwipBase with our parameters
        YagwipBase.__init__(self, gmx_path=gmx_path, debug=False)

        # Core file paths and state
        self.current_pdb_path: Optional[str] = None
        self.ligand_pdb_path: Optional[str] = None
        self.basename: Optional[str] = None
        self.current_ligand_name: Optional[str] = None

        # Print welcome banner
        self.print_banner()

        # Initialize component handlers
        self.user_itp_paths: List[str] = []
        self.editor = Editor()
        self.ligand_pipeline = LigandPipeline(logger=self.logger, debug=self.debug)
        self.gmx = GromacsCommands(gmx_path=self.gmx_path, debug=self.debug, logger=self.logger)
        self.builder = Builder(gmx_path=self.gmx_path, debug=self.debug, logger=self.logger)

        # Initialize runtime monitor
        from utils.log_utils import RuntimeMonitor
        self.runtime_monitor = RuntimeMonitor(logger=self.logger, debug_mode=self.debug)

        # Validate GROMACS installation
        try:
            validate_gromacs_installation(gmx_path)
        except RuntimeError as e:
            self._log_error(f"GROMACS Validation Error: {e}")
            self._log_error("YAGWIP cannot start without GROMACS. Please install GROMACS and try again.")
            sys.exit(1)

        # Custom command system
        self.custom_cmds: Dict[str, str] = {k: "" for k in ("pdb2gmx", "solvate", "genions")}
        self.ligand_counter: int = 0  # For FEP-style ligand naming

    def _setup(self) -> None:
        """Component-specific initialization. Override from YagwipBase."""
        # Additional setup can be added here if needed
        pass

    def _require_pdb(self) -> bool:
        """
        Check if a PDB file is loaded and available for processing.

        This method validates that a PDB file has been loaded before
        executing commands that require structural data.

        Returns:
            True if a PDB file is loaded, False otherwise

        Note:
            In debug mode, this check is bypassed to allow testing
            without actual PDB files.
        """
        if not self.current_pdb_path and not self.debug:
            self._log_error("No PDB loaded.")
            return False
        return True

    def onecmd(self, line: str) -> bool:
        """
        Override onecmd to log user input commands before execution.

        This method intercepts all user commands and logs them before
        passing them to the parent cmd.Cmd class for execution.

        Args:
            line: The command line input from the user

        Returns:
            bool: True if the command should exit the shell, False otherwise
        """
        # Log the user input command
        if line.strip():  # Only log non-empty commands
            self._log_info(f"[USER_COMMAND] {line}")

            # Start command monitoring with the user input
            from utils.log_utils import command_context
            with command_context(f"user_command: {line}", self.logger, self.debug, user_input=line) as cmd_metrics:
                # Execute the command using the parent class
                result = super().onecmd(line)

                # Collect operations from all components
                all_operations = []

                # Collect from main shell runtime monitor
                if hasattr(self, 'runtime_monitor') and self.runtime_monitor.metrics_history:
                    all_operations.extend(self.runtime_monitor.metrics_history)

                # Collect from builder
                if hasattr(self, 'builder') and hasattr(self.builder, 'runtime_monitor') and self.builder.runtime_monitor.metrics_history:
                    all_operations.extend(self.builder.runtime_monitor.metrics_history)

                # Collect from gmx
                if hasattr(self, 'gmx') and hasattr(self.gmx, 'runtime_monitor') and self.gmx.runtime_monitor.metrics_history:
                    all_operations.extend(self.gmx.runtime_monitor.metrics_history)

                # Update command metrics with all operations
                if cmd_metrics and all_operations:
                    cmd_metrics.sub_operations = all_operations
                    cmd_metrics.total_operations = len(all_operations)
                    cmd_metrics.successful_operations = sum(1 for op in all_operations if op.success)
                    cmd_metrics.failed_operations = cmd_metrics.total_operations - cmd_metrics.successful_operations

                # Store command metrics for runtime display
                if hasattr(self, 'runtime_monitor') and cmd_metrics:
                    self.runtime_monitor = cmd_metrics

                return result

        return super().onecmd(line)

    def default(self, line: str) -> None:
        """
        Handle unrecognized commands.

        This method is called when a command is not recognized by
        the shell. It provides a helpful error message to guide
        the user.

        Args:
            line: The unrecognized command line
        """
        self._log_error(f"Unknown command: {line}")

    def do_debug(self, arg: str) -> None:
        """
        Toggle debug mode on/off with enhanced logging and system monitoring.

        Debug mode provides detailed information about:
        - Command execution details
        - System resource statistics (CPU, memory, disk usage)
        - Runtime monitoring data
        - File operations and validation

        Commands are still executed in debug mode, but with verbose
        output for troubleshooting and development.

        Usage:
            debug          - Toggle debug mode
            debug on       - Enable debug mode
            debug off      - Disable debug mode

        Args:
            arg: Debug mode argument ('on', 'off', or empty for toggle)
        """
        arg = arg.lower().strip()
        if arg == "on":
            self.debug = True
        elif arg == "off":
            self.debug = False
        else:
            self.debug = not self.debug

        # Update logger with new debug mode
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

    def print_banner(self) -> None:
        """
        Print the YAGWIP banner logo on shell startup.

        Loads and displays the ASCII art banner from the assets
        directory. If the banner file cannot be loaded, the error
        is logged but the shell continues to function.
        """
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(os.path.dirname(module_dir), "assets")
            banner_path = os.path.join(assets_dir, "yagwip_banner.txt")
            with open(str(banner_path), "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            self._log_error(f"Could not load banner: {e}")

    def do_show(self, arg: str) -> None:
        """
        Display current custom or default command configurations.

        Shows the current state of custom command overrides for
        pdb2gmx, solvate, and genions commands. If no custom
        command is set, '[DEFAULT]' is displayed.

        Usage:
            show

        Args:
            arg: Command argument (unused)
        """
        for k in ["pdb2gmx", "solvate", "genions"]:
            cmd_str = self.custom_cmds.get(k)
            self._log_info(f"{k}: {cmd_str if cmd_str else '[DEFAULT]'}")

    def do_runtime(self, arg: str) -> None:
        """
        Display runtime statistics and performance metrics.

        Shows comprehensive runtime information including:
        - Total operations performed
        - Success/failure rates
        - Average operation duration
        - System resource usage patterns

        This information is useful for performance monitoring
        and debugging long-running workflows.

        Usage:
            runtime          # Show runtime statistics
            runtime clear    # Clear runtime history

        Args:
            arg: Command argument (unused or "clear")
        """
        if arg.strip().lower() == "clear":
            # Clear runtime history
            self.runtime_monitor.metrics_history.clear()
            if hasattr(self, 'builder') and hasattr(self.builder, 'runtime_monitor'):
                self.builder.runtime_monitor.metrics_history.clear()
            if hasattr(self, 'gmx') and hasattr(self.gmx, 'runtime_monitor'):
                self.gmx.runtime_monitor.metrics_history.clear()
            self._log_info("Runtime history cleared.")
            return

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

    def do_set(self, arg: str) -> None:
        """
        Edit default command strings for GROMACS tools.

        Allows customization of the default commands used for
        pdb2gmx, solvate, and genions. The user is shown the
        current command and can modify it inline.

        Usage:
            set pdb2gmx    - Edit pdb2gmx command
            set solvate    - Edit solvate command
            set genions    - Edit genions command

        Args:
            arg: The command type to edit ('pdb2gmx', 'solvate', or 'genions')

        Note:
            Press ENTER to accept the modified command.
            Type 'quit' to cancel the edit operation.
        """
        valid_keys = ["pdb2gmx", "solvate", "genions"]
        cmd_key = arg.strip().lower()
        if cmd_key not in valid_keys:
            self._log_error(f"Usage: set <{'|'.join(valid_keys)}>")
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
        self._log_info(f"Current command for {cmd_key}:\n{current}")
        self._log_info("Type new command or press ENTER to keep current. Type 'quit' to cancel.")
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

    def _complete_filename(self, text: str, suffix: str, line: Optional[str] = None,
                          begidx: Optional[int] = None, endidx: Optional[int] = None) -> List[str]:
        """
        Generic TAB autocomplete for filenames matching a specific suffix.

        Provides intelligent filename completion for commands that require
        specific file types (e.g., .pdb, .gro, .top files).

        Args:
            text: The current input text to match
            suffix: The file suffix or pattern to match (e.g., ".pdb", "solv.ions.gro")
            line: The complete command line (unused)
            begidx: Beginning index of the word being completed (unused)
            endidx: Ending index of the word being completed (unused)

        Returns:
            List of matching filenames in the current directory

        Example:
            If text="pro" and suffix=".pdb", returns ["protein.pdb", "protein_clean.pdb"]
        """
        if not text:
            return [f for f in os.listdir() if f.endswith(suffix)]
        return [f for f in os.listdir() if f.startswith(text) and f.endswith(suffix)]

    def complete_loadpdb(self, text: str, line: Optional[str] = None,
                        begidx: Optional[int] = None, endidx: Optional[int] = None) -> List[str]:
        """
        Tab completion for .pdb files in the loadpdb command.

        Args:
            text: The current input text to match
            line: The complete command line
            begidx: Beginning index of the word being completed
            endidx: Ending index of the word being completed

        Returns:
            List of .pdb files in the current directory
        """
        return self._complete_filename(text, ".pdb", line, begidx, endidx)

    def do_loadpdb(self, arg: str) -> None:
        """
        Load and process a PDB file for molecular dynamics simulation setup.

        This is the primary command for loading structural data into YAGWIP.
        The command automatically detects the type of system (protein-only,
        ligand-only, or protein-ligand complex) and processes it accordingly.

        For protein-ligand systems, the command:
        - Separates protein and ligand coordinates
        - Handles CONNECT records for bond information
        - Optionally runs ligand parameterization pipeline
        - Prepares files for GROMACS processing

        For ligand-only systems:
        - Processes ligand coordinates
        - Optionally runs quantum chemistry calculations
        - Generates force field parameters

        Usage:
            loadpdb <filename.pdb> [--ligand_builder] [--c CHARGE] [--m MULTIPLICITY]

        Arguments:
            filename.pdb: PDB file to load (required)
            --ligand_builder: Enable ligand parameterization pipeline (requires ORCA)
            --c CHARGE: Total charge for quantum chemistry calculations (default: 0)
            --m MULTIPLICITY: Spin multiplicity for quantum chemistry (default: 1)

        Examples:
            loadpdb protein.pdb                    # Load protein-only system
            loadpdb complex.pdb --ligand_builder   # Load complex with ligand parameterization
            loadpdb ligand.pdb --c 1 --m 2        # Load ligand with charge=1, multiplicity=2

        Note:
            The --ligand_builder option requires ORCA to be installed and available
            in the system PATH for quantum chemistry calculations.
        """
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("loadpdb_command")

        try:
            # Parse arguments
            args_op = self.runtime_monitor.start_operation("parse_loadpdb_args")
            args = self._parse_loadpdb_args(arg)
            self.runtime_monitor.end_operation(success=True)

            # Read PDB file
            read_op = self.runtime_monitor.start_operation("read_pdb_file")
            try:
                lines = self._read_pdb_file(args.pdb_file)
                self.runtime_monitor.end_operation(success=True)
            except FileNotFoundError:
                self.runtime_monitor.end_operation(success=False, error_message="PDB file not found")
                return

            # Split PDB lines
            split_op = self.runtime_monitor.start_operation("split_pdb_lines")
            hetatm_lines, atom_lines = self._split_pdb_lines(lines)
            self.runtime_monitor.end_operation(success=True)

            # Determine system type and handle accordingly
            if args.ligand_builder and not hetatm_lines:
                self._log_info("No Ligand Detected")
                handle_op = self.runtime_monitor.start_operation("handle_protein_only")
                self._handle_protein_only(lines)
                self.runtime_monitor.end_operation(success=True)
            elif hetatm_lines and not atom_lines:
                handle_op = self.runtime_monitor.start_operation("handle_ligand_only")
                self._handle_ligand_only(hetatm_lines, args)
                self.runtime_monitor.end_operation(success=True)
            elif hetatm_lines:
                handle_op = self.runtime_monitor.start_operation("handle_protein_ligand")
                self._handle_protein_ligand(lines, hetatm_lines, args)
                self.runtime_monitor.end_operation(success=True)
            else:
                handle_op = self.runtime_monitor.start_operation("handle_protein_only")
                self._handle_protein_only(lines)
                self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def _parse_loadpdb_args(self, arg: str) -> argparse.Namespace:
        """
        Parse command line arguments for the loadpdb command.

        Args:
            arg: Command line argument string

        Returns:
            Parsed arguments namespace

        Raises:
            SystemExit: If argument parsing fails
        """
        parser = argparse.ArgumentParser(description="Load PDB file")
        parser.add_argument("pdb_file", help="PDB file to load")
        parser.add_argument(
            "--ligand_builder", action="store_true", help="Use ligand builder"
        )
        parser.add_argument(
            "--c", type=int, default=0, help="Total charge for QM input"
        )
        parser.add_argument(
            "--m", type=int, default=1, help="Multiplicity for QM input"
        )
        # Use shlex.split to properly handle negative values and quoted strings
        return parser.parse_args(shlex.split(arg))

    def _read_pdb_file(self, pdb_file: str) -> List[str]:
        """
        Read and validate a PDB file.

        Loads the PDB file, validates its existence, and sets up
        the internal state for further processing.

        Args:
            pdb_file: Path to the PDB file

        Returns:
            List of lines from the PDB file

        Raises:
            FileNotFoundError: If the PDB file doesn't exist
        """
        full_path = os.path.abspath(pdb_file)
        if not os.path.isfile(full_path):
            self._log_error(f"'{pdb_file}' not found.")
            raise FileNotFoundError
        self.current_pdb_path = full_path
        self.basename = os.path.splitext(os.path.basename(full_path))[0]
        self._log_success(f"PDB file loaded: {full_path}")
        with open(full_path, "r") as f:
            return f.readlines()

    def _split_pdb_lines(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split PDB file lines into HETATM and ATOM records.

        Separates ligand atoms (HETATM records) from protein atoms
        (ATOM records) for independent processing.

        Args:
            lines: List of lines from the PDB file

        Returns:
            Tuple of (hetatm_lines, atom_lines) where each is a list of strings
        """
        hetatm_lines = [line for line in lines if line.startswith("HETATM")]
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        return hetatm_lines, atom_lines

    def _handle_ligand_only(self, hetatm_lines: List[str], args: argparse.Namespace) -> None:
        """
        Process a ligand-only PDB file.

        Handles the case where the PDB file contains only ligand atoms
        (HETATM records) without protein structure.

        Args:
            hetatm_lines: List of HETATM record lines
            args: Parsed command line arguments
        """
        ligand_name = self._assign_ligand_name()
        ligand_file = f"{ligand_name}.pdb"
        self.ligand_pdb_path = os.path.abspath(ligand_file)
        self._write_ligand_pdb(ligand_file, hetatm_lines)
        self._warn_if_no_hydrogens(hetatm_lines)
        itp_file = f"{ligand_name}.itp"
        if os.path.isfile(itp_file):
            self._process_ligand_itp(itp_file, ligand_name)
        elif args.ligand_builder:
            self._run_ligand_builder(ligand_file, ligand_name, args.c, args.m)
            if os.path.isfile(itp_file):
                self._process_ligand_itp(itp_file, ligand_name)
        else:
            self._log_info(f"{itp_file} not found and --ligand_builder not specified.")

    def _handle_protein_ligand(self, lines: List[str], hetatm_lines: List[str],
                              args: argparse.Namespace) -> None:
        """
        Process a protein-ligand complex PDB file.

        Handles the case where the PDB file contains both protein (ATOM)
        and ligand (HETATM) atoms. Separates them into individual files
        and processes the ligand if requested.

        Args:
            lines: All lines from the PDB file
            hetatm_lines: List of HETATM record lines
            args: Parsed command line arguments
        """
        ligand_name = self._assign_ligand_name()
        protein_file, ligand_file, connect_records = (
            self._extract_ligand_and_protein_with_connect(lines, ligand_name)
        )
        self.ligand_pdb_path = os.path.abspath(ligand_file)
        self._warn_if_no_hydrogens(hetatm_lines)
        itp_file = f"{ligand_name}.itp"
        if os.path.isfile(itp_file):
            self._process_ligand_itp(itp_file, ligand_name)
        elif args.ligand_builder:
            # Pass CONNECT records to ligand_pipeline
            self._run_ligand_builder(
                ligand_file,
                ligand_name,
                args.c,
                args.m,
                connect_records=connect_records,
            )
            if os.path.isfile(itp_file):
                self._process_ligand_itp(itp_file, ligand_name)
        else:
            self._log_info(f"{itp_file} not found and --ligand_builder not specified.")

    def _warn_if_missing_residues(self, protein_pdb: str) -> List[Tuple[str, int, int]]:
        """
        Identify missing internal residues by checking for gaps in residue numbering.

        Analyzes the protein PDB file to detect missing residues that could
        affect simulation quality. Reports gaps in residue numbering by chain.

        Args:
            protein_pdb: Path to the protein PDB file

        Returns:
            List of tuples containing (chain_id, current_residue, next_residue)
            for each gap found

        Note:
            This is a warning system - gaps don't prevent simulation setup
            but may indicate structural issues that should be addressed.
        """
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("warn_if_missing_residues")

        try:
            self._log_info(f"Checking for missing residues in {protein_pdb}")
            residue_map = {}  # {chain_id: sorted list of residue IDs}

            # Read PDB file
            read_op = self.runtime_monitor.start_operation("read_pdb_for_residue_check")
            with open(protein_pdb, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        chain_id = line[21].strip() or "A"
                        try:
                            res_id = int(line[22:26].strip())
                        except ValueError:
                            continue
                        if chain_id not in residue_map:
                            residue_map[chain_id] = set()
                        residue_map[chain_id].add(res_id)
            self.runtime_monitor.end_operation(success=True)

            # Analyze gaps
            analyze_op = self.runtime_monitor.start_operation("analyze_residue_gaps")
            gaps = []
            for chain_id, residues in residue_map.items():
                sorted_residues = sorted(residues)
                for i in range(len(sorted_residues) - 1):
                    current = sorted_residues[i]
                    next_expected = current + 1
                    if sorted_residues[i + 1] != next_expected:
                        gaps.append((chain_id, current, sorted_residues[i + 1]))
            self.runtime_monitor.end_operation(success=True)

            if gaps:
                self._log_warning(f"Found missing residue ranges: {gaps}")
            else:
                self._log_info("No gaps found.")

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

        return gaps

    def _handle_protein_only(self, lines: List[str]) -> None:
        """
        Process a protein-only PDB file.

        Handles the case where the PDB file contains only protein atoms
        (ATOM records) without any ligands.

        Args:
            lines: List of lines from the PDB file
        """
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("handle_protein_only")

        try:
            self.ligand_pdb_path = None

            # Write protein PDB
            write_op = self.runtime_monitor.start_operation("write_protein_pdb")
            with open("protein.pdb", "w", encoding="utf-8") as prot_out:
                for line in lines:
                    if line[17:20] in ("HSP", "HSD"):
                        line = line[:17] + "HIS" + line[20:]
                    prot_out.write(line)
            self.runtime_monitor.end_operation(success=True)

            self._log_info(
                "No HETATM entries found. Wrote corrected PDB to protein.pdb and using it as apo protein."
            )

            # Check for missing residues
            check_op = self.runtime_monitor.start_operation("check_missing_residues")
            self._warn_if_missing_residues(protein_pdb="protein.pdb")
            self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def _assign_ligand_name(self) -> str:
        """
        Assign a unique name to the current ligand.

        Uses a counter to generate sequential ligand names (ligandA, ligandB, etc.)
        for FEP-style workflows.

        Returns:
            Unique ligand name string
        """
        ligand_name = chr(ord("A") + self.ligand_counter)
        ligand_name = f"ligand{ligand_name}"
        self.current_ligand_name = ligand_name
        self.ligand_counter += 1
        return ligand_name

    def _write_ligand_pdb(self, ligand_file: str, hetatm_lines: List[str]) -> None:
        """
        Write ligand atoms to a separate PDB file.

        Creates a new PDB file containing only the ligand atoms,
        with the residue name changed to "LIG" for consistency.

        Args:
            ligand_file: Output filename for the ligand PDB
            hetatm_lines: List of HETATM record lines
        """
        with open(ligand_file, "w", encoding="utf-8") as lig_out:
            for line in hetatm_lines:
                lig_out.write(line[:17] + "LIG" + line[20:])
        self._log_info(
            f"Ligand-only PDB detected. Assigned name: {ligand_file}. Wrote ligand to {ligand_file}"
        )

    def _warn_if_no_hydrogens(self, hetatm_lines: List[str]) -> None:
        """
        Check if ligand atoms include hydrogen atoms and warn if missing.

        Analyzes HETATM records to detect the presence of hydrogen atoms.
        Missing hydrogens can affect force field assignment and simulation accuracy.

        Args:
            hetatm_lines: List of HETATM record lines
        """
        if not any(
            line[76:78].strip() == "H" or line[12:16].strip().startswith("H")
            for line in hetatm_lines
        ):
            self._log_warning(
                "Ligand appears to lack hydrogen atoms. Consider checking hydrogens and valences."
            )

    def _run_ligand_builder(
        self, ligand_file: str, ligand_name: str, charge: int, multiplicity: int,
        connect_records: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """
        Execute the complete ligand parameterization pipeline.

        This method orchestrates the full ligand parameterization workflow:
        1. Copies AMBER force field files
        2. Converts PDB to MOL2 format with bond detection
        3. Runs quantum chemistry calculations with ORCA
        4. Applies calculated charges to MOL2 file
        5. Generates GROMACS topology with ACPYPE

        Args:
            ligand_file: Path to the ligand PDB file
            ligand_name: Name assigned to the ligand
            charge: Total charge for quantum chemistry calculations
            multiplicity: Spin multiplicity for quantum chemistry
            connect_records: Optional CONNECT records from PDB for bond information

        Note:
            This method requires ORCA to be installed and available in the system PATH.
            The workflow generates several intermediate files and the final GROMACS
            topology files needed for simulation.
        """
        # Copy AMBER force field files
        amber_ff_source = str(files("templates").joinpath("amber14sb.ff/"))
        amber_ff_dest = os.path.abspath("amber14sb.ff")
        if not os.path.exists(amber_ff_dest):
            os.makedirs(amber_ff_dest)
            self._log_info(f"Created directory: {amber_ff_dest}")
            try:
                for item in Path(amber_ff_source).iterdir():
                    if item.is_file():
                        content = item.read_text(encoding="utf-8")
                        dest_file = os.path.join(amber_ff_dest, item.name)
                        with open(dest_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        self._log_debug(f"Copied {item.name}")
                self._log_success("Copied all amber14sb.ff files.")
            except Exception as e:
                self._log_error(f"Failed to copy amber14sb.ff files: {e}")
        else:
            self._log_info(f"amber14sb.ff already exists, not overwriting.")

        # Convert PDB to MOL2 with bond detection
        mol2_file = self.ligand_pipeline.convert_pdb_to_mol2(
            ligand_file, connect_records=connect_records
        )
        if not mol2_file or not os.path.isfile(mol2_file):
            self._log_error(
                f"MOL2 generation failed or file not found: {mol2_file}. Aborting ligand pipeline..."
            )
            return

        # Extract atom information for ORCA input
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
            self._log_error("Could not find ATOM section in MOL2 file.")
            return
        if atom_end is None:
            atom_end = len(lines)
        atom_lines = lines[atom_start:atom_end]

        # Create pandas DataFrame for atom data
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

        # Generate ORCA input file
        orca_geom_input = mol2_file.replace(".mol2", ".inp")
        self.ligand_pipeline.mol2_dataframe_to_orca_charge_input(
            df_atoms,
            orca_geom_input,
            charge=charge,
            multiplicity=multiplicity,
        )

        # Run ORCA calculation
        self.ligand_pipeline.run_orca(orca_geom_input)

        # Check if ORCA property file exists before applying charges
        property_file = f"orca/{ligand_name}.property.txt"
        if not os.path.exists(property_file):
            self._log_error(
                f"ORCA property file not found: {property_file}. ORCA calculation may have failed."
            )
            return

        # Apply calculated charges and generate topology
        self.ligand_pipeline.apply_orca_charges_to_mol2(mol2_file, property_file)
        self.ligand_pipeline.run_acpype(mol2_file, charge=charge, multiplicity=multiplicity)
        self.ligand_pipeline.copy_acpype_output_files(mol2_file)

    def _process_ligand_itp(self, itp_file: str, ligand_name: str) -> None:
        """
        Process an existing ligand ITP file for GROMACS compatibility.

        Integrates the ligand topology into the force field system by:
        1. Appending atom types to the force field
        2. Modifying improper dihedrals if needed
        3. Renaming residues for consistency

        Args:
            itp_file: Path to the ligand ITP file
            ligand_name: Name of the ligand for force field integration
        """
        self.editor.append_ligand_atomtypes_to_forcefield(itp_file, ligand_name)
        self.editor.ligand_itp = itp_file
        self.editor.modify_improper_dihedrals_in_ligand_itp()
        self.editor.rename_residue_in_itp_atoms_section()

    def _extract_ligand_and_protein_with_connect(self, lines: List[str], ligand_name: str) -> Tuple[str, str, Dict[int, List[int]]]:
        """
        Extract ligand and protein coordinates from a complex PDB file.

        Separates protein and ligand atoms from a complex PDB file while
        preserving CONNECT records for bond information. This is essential
        for maintaining proper bond connectivity in the ligand.

        Args:
            lines: All lines from the complex PDB file
            ligand_name: Name to assign to the ligand

        Returns:
            Tuple containing:
            - protein_file: Path to the extracted protein PDB file
            - ligand_file: Path to the extracted ligand PDB file
            - connect_records: Dictionary mapping atom indices to bonded atoms
        """
        protein_file = "protein.pdb"
        ligand_file = f"{ligand_name}.pdb"
        ligand_indices = []
        connect_records = {}

        # Extract protein and ligand atoms
        with open(protein_file, "w", encoding="utf-8") as prot_out, open(
            ligand_file, "w", encoding="utf-8"
        ) as lig_out:
            for line in lines:
                if line.startswith("HETATM"):
                    lig_out.write(line[:17] + "LIG" + line[20:])
                    try:
                        idx = int(line[6:11])
                        ligand_indices.append(idx)
                    except Exception:
                        pass
                else:
                    if line[17:20] in ("HSP", "HSD"):
                        line = line[:17] + "HIS" + line[20:]
                    prot_out.write(line)

        # Extract CONNECT records for ligand atoms
        for line in lines:
            if line.startswith("CONECT"):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        idx = int(parts[1])
                        if idx in ligand_indices:
                            bonded = [
                                int(x)
                                for x in parts[2:]
                                if x.isdigit() and int(x) in ligand_indices
                            ]
                            if bonded:
                                connect_records[idx] = bonded
                    except Exception:
                        pass

        self._log_info(
            f"Detected ligand. Split into: {protein_file}, {ligand_file}, with {len(connect_records)} ligand CONNECT records."
        )
        return protein_file, ligand_file, connect_records

    def do_fep_prep(self, arg: str) -> None:
        """
        Run the Free Energy Perturbation (FEP) preparation workflow.

        This command orchestrates the complete FEP setup process, which includes:
        1. Finding Maximum Common Substructure (MCS) between ligands
        2. Writing atom mapping file (atom_map.txt)
        3. Aligning ligandB structures to ligandA structures
        4. Creating hybrid topology for FEP simulations
        5. Organizing files into A_to_B and B_to_A directories

        The workflow requires both ligand A and ligand B files to be present
        in the current directory.

        Usage:
            fep_prep

        Required Files:
            - ligandA.mol2, ligandB.mol2: MOL2 files for both ligands
            - ligandA.pdb, ligandB.pdb: PDB files for both ligands
            - ligandA.gro, ligandB.gro: GROMACS coordinate files
            - ligandA.itp, ligandB.itp: GROMACS topology files

        Output:
            - atom_map.txt: Atom mapping between ligands
            - ligandB_aligned.mol2/pdb/gro: Aligned ligand B structures
            - hybrid.itp: Hybrid topology for FEP
            - hybrid_stateA.pdb/gro: Hybrid state A structures
            - hybrid_stateB.pdb/gro: Hybrid state B structures
            - Directory structure for FEP simulations

        Note:
            This command calls the external fep_prep.py script to perform
            the actual FEP preparation calculations.
        """
        cwd = os.getcwd()
        required_files = [
            "ligandA.mol2",
            "ligandB.mol2",
            "ligandA.pdb",
            "ligandA.gro",
            "ligandA.itp",
            "ligandB.pdb",
            "ligandB.gro",
            "ligandB.itp",
        ]
        missing = [
            f for f in required_files if not os.path.isfile(os.path.join(cwd, f))
        ]
        if missing:
            self._log_error(f"Missing required files: {', '.join(missing)}")
            return

        # Prepare command to call fep_prep.py
        fep_prep_path = os.path.join(os.path.dirname(__file__), "fep_prep.py")
        python_exe = sys.executable
        cmd = [
            python_exe,
            fep_prep_path,
            "--ligA_mol2",
            "ligandA.mol2",
            "--ligB_mol2",
            "ligandB.mol2",
            "--ligA_pdb",
            "ligandA.pdb",
            "--ligA_gro",
            "ligandA.gro",
            "--ligA_itp",
            "ligandA.itp",
            "--ligB_pdb",
            "ligandB.pdb",
            "--ligB_gro",
            "ligandB.gro",
            "--ligB_itp",
            "ligandB.itp",
            "--create_hybrid",  # Add hybrid topology creation
        ]

        # Log workflow steps
        self._log_info("FEP prep workflow:")
        self._log_info("  1. Find MCS and write atom_map.txt")
        self._log_info("  2. Align ligandB.mol2 to ligandA.mol2")
        self._log_info("  3. Align ligandB.pdb to ligandA.pdb")
        self._log_info("  4. Align ligandB.gro to ligandA.gro")
        self._log_info("  5. Create hybrid topology for FEP simulations")
        self._log_info("  6. Organize files into ligand_only/ and protein_complex/ directories")

        # Execute FEP preparation
        self._log_info(f"Running FEP prep: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._log_success("FEP preparation complete.")
            self._log_info(
                "Output files: atom_map.txt, ligandB_aligned.mol2, ligandB_aligned.pdb, ligandB_aligned.gro, hybrid.itp, hybrid_stateA.pdb, hybrid_stateB.pdb"
            )
            self._log_info("Directory structure:")
            self._log_info("  ligand_only/")
            self._log_info("    A_to_B/ - hybrid_stateA.gro, hybrid.itp")
            self._log_info("    B_to_A/ - hybrid_stateB.gro, hybrid.itp")
            self._log_info("  protein_complex/")
            self._log_info("    A_to_B/ - hybrid_stateA.pdb, protein.pdb, hybrid.itp")
            self._log_info("    B_to_A/ - hybrid_stateB.pdb, protein.pdb, hybrid.itp")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            self._log_error(f"FEP prep failed: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

    def do_pdb2gmx(self, arg):
        """
        Run pdb2gmx. Handles protein-only, protein-ligand, and FEP (A/B) cases.
        """
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("pdb2gmx_command")

        try:
            # Check for protein.pdb
            check_op = self.runtime_monitor.start_operation("check_protein_pdb")
            if not os.path.isfile("protein.pdb"):
                self._log_info("protein.pdb not found. Switching to ligand-only workflow.")
                self.runtime_monitor.end_operation(success=True)
                # Call _pdb2gmx_ligand with dummy args (lambda_dirs=None, output_gro=None) for now
                ligand_op = self.runtime_monitor.start_operation("pdb2gmx_ligand_only")
                self._pdb2gmx_ligand()
                self.runtime_monitor.end_operation(success=True)
                return
            self.runtime_monitor.end_operation(success=True)

            # Setup amber14sb.ff
            amber_op = self.runtime_monitor.start_operation("setup_amber_forcefield")
            amber_ff_source = str(files("templates").joinpath("amber14sb.ff/"))
            amber_ff_dest = os.path.abspath("amber14sb.ff")
            if not os.path.exists(amber_ff_dest):
                os.makedirs(amber_ff_dest)
                self._log_info(f"Created directory: {amber_ff_dest}")
                try:
                    for item in Path(amber_ff_source).iterdir():
                        if item.is_file():
                            content = item.read_text(encoding="utf-8")
                            dest_file = os.path.join(amber_ff_dest, item.name)
                            with open(dest_file, "w", encoding="utf-8") as f:
                                f.write(content)
                            self._log_debug(f"Copied {item.name}")
                    self._log_success("Copied all amber14sb.ff files.")
                except Exception as e:
                    self._log_error(f"Failed to copy amber14sb.ff files: {e}")
                    self.runtime_monitor.end_operation(success=False, error_message=str(e))
                    return
            else:
                self._log_info(f"amber14sb.ff already exists, not overwriting.")
            self.runtime_monitor.end_operation(success=True)

            # Run pdb2gmx on protein
            pdb2gmx_op = self.runtime_monitor.start_operation("run_pdb2gmx_protein")
            protein_pdb = "protein"
            output_gro = f"{protein_pdb}.gro"
            self.builder.run_pdb2gmx(
                protein_pdb, custom_command=self.custom_cmds.get("pdb2gmx")
            )
            if not os.path.isfile(output_gro):
                self._log_error(f"Expected {output_gro} was not created by pdb2gmx.")
                self.runtime_monitor.end_operation(success=False, error_message=f"Expected {output_gro} was not created")
                return
            self.runtime_monitor.end_operation(success=True)

            # Check for FEP directories
            fep_check_op = self.runtime_monitor.start_operation("check_fep_directories")
            fep_dirs = ["ligand_only", "protein_complex"]
            fep_present = all(os.path.isdir(d) for d in fep_dirs)
            self.runtime_monitor.end_operation(success=True)

            if fep_present:
                fep_op = self.runtime_monitor.start_operation("pdb2gmx_fep_workflow")
                self._pdb2gmx_fep(fep_dirs)
                self.runtime_monitor.end_operation(success=True)
            else:
                # Check for ligand presence
                ligand_check_op = self.runtime_monitor.start_operation("check_ligand_presence")
                ligand_present = os.path.isfile("ligandA.pdb") or os.path.isfile("ligand.pdb")
                self.runtime_monitor.end_operation(success=True)

                if ligand_present:
                    protein_ligand_op = self.runtime_monitor.start_operation("pdb2gmx_protein_ligand")
                    self._pdb2gmx_protein_ligand(output_gro)
                    self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def _pdb2gmx_fep(self, fep_dirs):
        """
        Run pdb2gmx for FEP workflow (ligand_only and protein_complex directories at A_to_B and B_to_A level).
        """
        self._log_info("Detected FEP directories: {}".format(", ".join(fep_dirs)))

        # Copy amber14sb.ff from current working directory to all FEP directories
        amber_ff_source = "amber14sb.ff"

        # Check if amber14sb.ff exists in current working directory
        if not os.path.exists(amber_ff_source):
            self._log_error(
                f"amber14sb.ff not found in current working directory: {amber_ff_source}"
            )
            return

        # Define the 4 FEP directories that need amber14sb.ff
        fep_amber_dirs = [
            os.path.join("ligand_only", "A_to_B"),
            os.path.join("ligand_only", "B_to_A"),
            os.path.join("protein_complex", "A_to_B"),
            os.path.join("protein_complex", "B_to_A"),
        ]

        # Copy amber14sb.ff to each FEP directory
        for fep_dir in fep_amber_dirs:
            if os.path.isdir(fep_dir):
                amber_ff_dest = os.path.join(fep_dir, "amber14sb.ff")
                if not os.path.exists(amber_ff_dest):
                    self._log_info(f"Copying amber14sb.ff to {fep_dir}")
                    try:
                        # Use shutil.copytree to copy the entire directory
                        shutil.copytree(amber_ff_source, amber_ff_dest)
                        self._log_success(f"Copied amber14sb.ff to {fep_dir}")
                    except Exception as e:
                        self._log_error(
                            f"Failed to copy amber14sb.ff to {fep_dir}: {e}"
                        )
                else:
                    self._log_info(
                        f"amber14sb.ff already exists in {fep_dir}, not overwriting"
                    )

        self._log_info("Detected FEP directories: {}".format(", ".join(fep_dirs)))

        # Process hybrid.itp files in each FEP directory
        for fep_dir in fep_amber_dirs:
            if os.path.isdir(fep_dir):
                hybrid_itp_path = os.path.join(fep_dir, "hybrid.itp")
                if os.path.exists(hybrid_itp_path):
                    self._log_info(f"Processing hybrid.itp in {fep_dir}")
                    try:
                        # Create a new Editor instance for this FEP directory
                        # This ensures the Editor uses the correct paths for this context
                        fep_editor = Editor(
                            ligand_itp=hybrid_itp_path,
                            ffnonbonded_itp=os.path.join(
                                fep_dir, "amber14sb.ff", "ffnonbonded.itp"
                            ),
                        )

                        # Process the hybrid.itp with the FEP-specific Editor
                        fep_editor.append_ligand_atomtypes_to_forcefield(
                            hybrid_itp_path, "LIG"
                        )
                        fep_editor.ligand_itp = hybrid_itp_path
                        fep_editor.modify_improper_dihedrals_in_ligand_itp()
                        fep_editor.rename_residue_in_itp_atoms_section()

                        self._log_success(f"Processed hybrid.itp in {fep_dir}")
                    except Exception as e:
                        self._log_error(
                            f"Failed to process hybrid.itp in {fep_dir}: {e}"
                        )
                else:
                    self._log_warning(f"hybrid.itp not found in {fep_dir}")

        # Process ligand_only directories
        ligand_only_dir = "ligand_only"
        if os.path.isdir(ligand_only_dir):
            self._log_info(f"Processing {ligand_only_dir} directories...")

            # Process A_to_B directory
            a_to_b_dir = os.path.join(ligand_only_dir, "A_to_B")
            if os.path.isdir(a_to_b_dir):
                self._log_info(f"Processing {a_to_b_dir}...")
                # Change to A_to_B directory and run _pdb2gmx_ligand
                original_cwd = os.getcwd()
                try:
                    os.chdir(a_to_b_dir)
                    self._pdb2gmx_ligand()
                finally:
                    os.chdir(original_cwd)

            # Process B_to_A directory
            b_to_a_dir = os.path.join(ligand_only_dir, "B_to_A")
            if os.path.isdir(b_to_a_dir):
                self._log_info(f"Processing {b_to_a_dir}...")
                # Change to B_to_A directory and run _pdb2gmx_ligand
                original_cwd = os.getcwd()
                try:
                    os.chdir(b_to_a_dir)
                    self._pdb2gmx_ligand()
                finally:
                    os.chdir(original_cwd)

        # Process protein_complex directories
        protein_complex_dir = "protein_complex"
        if os.path.isdir(protein_complex_dir):
            self._log_info(f"Processing {protein_complex_dir} directories...")

            # Check if protein.gro and topol.top exist in current working directory
            protein_gro_source = "protein.gro"
            topol_top_source = "topol.top"
            if not os.path.exists(protein_gro_source):
                self._log_error(
                    f"protein.gro not found in current directory: {protein_gro_source}"
                )
                return
            if not os.path.exists(topol_top_source):
                self._log_error(
                    f"topol.top not found in current directory: {topol_top_source}"
                )
                return

            # Process A_to_B directory
            a_to_b_dir = os.path.join(protein_complex_dir, "A_to_B")
            if os.path.isdir(a_to_b_dir):
                self._log_info(f"Processing {a_to_b_dir}...")
                # Copy protein.gro to A_to_B directory
                protein_gro_dest = os.path.join(a_to_b_dir, "protein.gro")
                shutil.copy2(protein_gro_source, protein_gro_dest)
                self._log_info(f"Copied protein.gro to {a_to_b_dir}")

                # Copy topol.top to A_to_B directory
                topol_top_dest = os.path.join(a_to_b_dir, "topol.top")
                shutil.copy2(topol_top_source, topol_top_dest)
                self._log_info(f"Copied topol.top to {a_to_b_dir}")

                self._log_info(f"Running pdb2gmx for {a_to_b_dir}")
                # Change to A_to_B directory and run _pdb2gmx_protein_ligand
                original_cwd = os.getcwd()
                try:
                    os.chdir(a_to_b_dir)
                    self._pdb2gmx_protein_ligand("protein.gro")
                finally:
                    os.chdir(original_cwd)

            # Process B_to_A directory
            b_to_a_dir = os.path.join(protein_complex_dir, "B_to_A")
            if os.path.isdir(b_to_a_dir):
                self._log_info(f"Processing {b_to_a_dir}...")
                # Copy protein.gro to B_to_A directory
                protein_gro_dest = os.path.join(b_to_a_dir, "protein.gro")
                shutil.copy2(protein_gro_source, protein_gro_dest)
                self._log_info(f"Copied protein.gro to {b_to_a_dir}")

                # Copy topol.top to B_to_A directory
                topol_top_dest = os.path.join(b_to_a_dir, "topol.top")
                shutil.copy2(topol_top_source, topol_top_dest)
                self._log_info(f"Copied topol.top to {b_to_a_dir}")

                self._log_info(f"Running pdb2gmx for {b_to_a_dir}")
                # Change to B_to_A directory and run _pdb2gmx_protein_ligand
                original_cwd = os.getcwd()
                try:
                    os.chdir(b_to_a_dir)
                    self._pdb2gmx_protein_ligand("protein.gro")
                finally:
                    os.chdir(original_cwd)

        self._log_success(
            "FEP pdb2gmx workflow completed for A_to_B and B_to_A directories."
        )

    def _pdb2gmx_protein_ligand(self, protein_gro):
        # Determine ligand PDB file (regular ligand or hybrid)
        ligand_pdb_file = None
        ligand_itp_file = None

        # Check for hybrid files first (FEP case)
        hybrid_pdb_files = ["hybrid_stateA.pdb", "hybrid_stateB.pdb"]
        for hybrid_file in hybrid_pdb_files:
            if os.path.exists(hybrid_file) and os.path.getsize(hybrid_file) > 0:
                ligand_pdb_file = hybrid_file
                ligand_itp_file = "hybrid.itp"
                self._log_info(f"Found hybrid file: {ligand_pdb_file}")
                break

        # If no hybrid files, check for regular ligand files
        if ligand_pdb_file is None:
            ligand_pdb_files = [f"ligand{c}.pdb" for c in string.ascii_uppercase]
            for fname in ligand_pdb_files:
                if os.path.exists(fname) and os.path.getsize(fname) > 0:
                    ligand_pdb_file = fname
                    ligand_itp_file = fname.replace(".pdb", ".itp")
                    self._log_info(f"Found regular ligand file: {ligand_pdb_file}")
                    break

        # Check if we found any ligand file
        if ligand_pdb_file is None:
            self._log_error(
                "No ligand PDB file found. Expected hybrid_stateA/B.pdb or ligandX.pdb"
            )
            return

        # Check if corresponding ITP file exists
        if ligand_itp_file is None or not os.path.exists(ligand_itp_file):
            self._log_error(f"Ligand ITP file not found: {ligand_itp_file}")
            return

        # Protein + ligand case
        if (
            ligand_pdb_file is not None
            and os.path.exists(ligand_pdb_file)
            and os.path.getsize(ligand_pdb_file) > 0
        ):
            self._log_info(
                f"Processing protein-ligand system with {ligand_pdb_file}..."
            )
            # Add ligand coordinates to protein gro and update topology
            self.editor.append_ligand_coordinates_to_gro(
                protein_gro, ligand_pdb_file, ligand_itp_file, "complex.gro"
            )
            self.editor.include_ligand_itp_in_topol(
                "topol.top", "LIG", ligand_itp_path=ligand_itp_file
            )

    def _pdb2gmx_ligand(self):
        """
        Handle the lambda directory workflow for pdb2gmx. Checks for ligandX.pdb in the current directory.
        If hybrid files are found, create topol.top for FEP workflow.
        """
        # Check for hybrid files first (FEP case)
        hybrid_gro_files = ["hybrid_stateA.gro", "hybrid_stateB.gro"]
        hybrid_found = False
        for hybrid_file in hybrid_gro_files:
            if os.path.isfile(hybrid_file):
                hybrid_found = True
                self._log_info(
                    f"Found hybrid file {hybrid_file} - processing FEP workflow"
                )
                break

        if hybrid_found:
            # For FEP case, the topol.top template already includes hybrid.itp
            if os.path.exists("hybrid.itp") and os.path.exists("topol.top"):
                self._log_info(
                    "FEP workflow detected - topol.top template already includes hybrid.itp"
                )
                self._log_success("topol.top is ready for FEP workflow")
            else:
                self._log_warning("hybrid.itp or topol.top not found for FEP workflow")
            return

        # If no hybrid files, proceed with regular ligand processing
        ligand_pdb_files = [f"ligand{c}.pdb" for c in string.ascii_uppercase]
        found = False
        base = None
        for fname in ligand_pdb_files:
            if os.path.isfile(fname):
                found = True
                base = fname
                break
        if not found:
            self._log_error(
                f"No ligand_*.pdb file found in current directory. Expected one of: {', '.join(ligand_pdb_files)}"
            )
            return
        # Find the first ligandX.acpype directory and copy the topology
        for c in string.ascii_uppercase:
            acpype_dir = f"ligand{c}.acpype"
            if os.path.isdir(acpype_dir):
                gmx_top = os.path.join(acpype_dir, f"ligand{c}_GMX.top")
                if os.path.isfile(gmx_top):
                    # Read the topology file
                    with open(gmx_top, "r") as f:
                        content = f.read()

                    # Remove the [ defaults ] block
                    content = re.sub(
                        r"\[ defaults \]\s*\n; nbfunc\s+comb-rule\s+gen-pairs\s+fudgeLJ fudgeQQ\s*\n\d+\s+\d+\s+\w+\s+[\d\.]+\s+[\d\.]+\s*\n",
                        "",
                        content,
                    )

                    # Remove the entire POSRES_LIG block
                    content = re.sub(
                        r'; Ligand position restraints\s*\n#ifdef POSRES_LIG\s*\n#include "posre_[^"]*\.itp"\s*\n#endif\s*\n',
                        "",
                        content,
                    )

                    # Remove all "_GMX" strings
                    content = content.replace("_GMX", "")

                    # Replace ligandX with LIG in the [ molecules ] section
                    content = re.sub(
                        r"ligand[A-Z]\s+\d+", "LIG              1", content
                    )

                    # Replace the ligand GMX itp include line with forcefield include
                    content = re.sub(
                        r'#include "ligand[A-Z]\.itp"\s*\n',
                        '#include "./amber14sb.ff/forcefield.itp"\n'
                        '; Include water topology\n#include "./amber14sb.ff/spce.itp"\n'
                        '; Include topology for ions\n#include "./amber14sb.ff/ions.itp"\n',
                        content,
                    )

                    # Write the modified content to topol.top
                    with open("topol.top", "w") as f:
                        f.write(content)

                    self._log_success(f"Modified and copied {gmx_top} to topol.top")
                    self.editor.include_ligand_itp_in_topol(
                        "topol.top", "LIG", ligand_itp_path=None
                    )
                else:
                    self._log_error(
                        "No ligand.acpype directory with ligand_GMX.top found."
                    )
                    return

    def do_solvate(self, arg):
        """
        Run solvate with optional custom command override. Handles three cases:
        1) Protein-only: solvates protein.gro
        2) Protein + single ligand: solvates complex.gro
        3) Ligand-only: solvates ligandX.gro in current directory
        4) FEP lambda subdirectories: solvates files in each lambda directory
        Usage: "solvate"
        Other Options: use "set solvate" to override defaults
        """
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("solvate_command")

        try:
            # Check for FEP directories
            fep_check_op = self.runtime_monitor.start_operation("check_fep_directories_solvate")
            fep_dirs = ["ligand_only", "protein_complex"]
            fep_present = all(os.path.isdir(d) for d in fep_dirs)
            self.runtime_monitor.end_operation(success=True)

            if fep_present:
                fep_op = self.runtime_monitor.start_operation("solvate_fep_workflow")
                self._do_solvate_fep(fep_dirs)
                self.runtime_monitor.end_operation(success=True)
            else:
                # Regular workflow
                base_check_op = self.runtime_monitor.start_operation("determine_solvate_base")
                base = self._determine_solvate_base()
                self.runtime_monitor.end_operation(success=True)

                if base is None:
                    self.runtime_monitor.end_operation(success=False, error_message="No suitable base found for solvation")
                    return

                pdb_check_op = self.runtime_monitor.start_operation("require_pdb_check")
                if not self._require_pdb():
                    self.runtime_monitor.end_operation(success=False, error_message="PDB requirement not met")
                    return
                self.runtime_monitor.end_operation(success=True)

                solvate_op = self.runtime_monitor.start_operation("run_solvate")
                self.builder.run_solvate(
                    base, custom_command=self.custom_cmds.get("solvate")
                )
                self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def _determine_solvate_base(self):
        """
        Determine which system to solvate based on available files.
        Returns the base name for solvation or None if no suitable files found.
        """
        if os.path.isfile("complex.gro"):
            return "complex"
        elif not os.path.isfile("protein.gro"):
            # Ligand-only: look for ligandX.gro or hybrid files (FEP case)
            ligand_gro_files = [f"ligand{c}.gro" for c in string.ascii_uppercase]
            hybrid_gro_files = ["hybrid_stateA.gro", "hybrid_stateB.gro"]

            # Check for hybrid files first (FEP case)
            for fname in hybrid_gro_files:
                if os.path.isfile(fname):
                    found = fname[:-4]  # strip .gro
                    return found

            # Check for regular ligand files
            found = None
            for fname in ligand_gro_files:
                if os.path.isfile(fname):
                    found = fname[:-4]  # strip .gro
                    break
            if found:
                return found
            else:
                self._log_error(
                    "No protein.gro, ligand_*.gro, or hybrid_state*.gro found for solvation."
                )
                return None
        else:
            return "protein"

    def _do_solvate_fep(self, fep_dirs):
        """
        Run solvate for FEP workflow (ligand_only and protein_complex directories at A_to_B and B_to_A level).
        """
        self._log_info("Detected FEP directories: {}".format(", ".join(fep_dirs)))

        # Process ligand_only directories
        ligand_only_dir = "ligand_only"
        if os.path.isdir(ligand_only_dir):
            self._log_info(f"Processing {ligand_only_dir} directories...")

            # Process A_to_B directory
            a_to_b_dir = os.path.join(ligand_only_dir, "A_to_B")
            if os.path.isdir(a_to_b_dir):
                self._log_info(f"Processing {a_to_b_dir}...")
                # Change to A_to_B directory and run solvate
                original_cwd = os.getcwd()
                try:
                    os.chdir(a_to_b_dir)
                    base = self._determine_solvate_base()
                    if base is not None:
                        self.builder.run_solvate(
                            base, custom_command=self.custom_cmds.get("solvate")
                        )
                finally:
                    os.chdir(original_cwd)

            # Process B_to_A directory
            b_to_a_dir = os.path.join(ligand_only_dir, "B_to_A")
            if os.path.isdir(b_to_a_dir):
                self._log_info(f"Processing {b_to_a_dir}...")
                # Change to B_to_A directory and run solvate
                original_cwd = os.getcwd()
                try:
                    os.chdir(b_to_a_dir)
                    base = self._determine_solvate_base()
                    if base is not None:
                        self.builder.run_solvate(
                            base, custom_command=self.custom_cmds.get("solvate")
                        )
                finally:
                    os.chdir(original_cwd)

        # Process protein_complex directories
        protein_complex_dir = "protein_complex"
        if os.path.isdir(protein_complex_dir):
            self._log_info(f"Processing {protein_complex_dir} directories...")

            # Process A_to_B directory
            a_to_b_dir = os.path.join(protein_complex_dir, "A_to_B")
            if os.path.isdir(a_to_b_dir):
                self._log_info(f"Processing {a_to_b_dir}...")
                # Change to A_to_B directory and run solvate
                original_cwd = os.getcwd()
                try:
                    os.chdir(a_to_b_dir)
                    base = self._determine_solvate_base()
                    if base is not None:
                        self.builder.run_solvate(
                            base, custom_command=self.custom_cmds.get("solvate")
                        )
                finally:
                    os.chdir(original_cwd)

            # Process B_to_A directory
            b_to_a_dir = os.path.join(protein_complex_dir, "B_to_A")
            if os.path.isdir(b_to_a_dir):
                self._log_info(f"Processing {b_to_a_dir}...")
                # Change to B_to_A directory and run solvate
                original_cwd = os.getcwd()
                try:
                    os.chdir(b_to_a_dir)
                    base = self._determine_solvate_base()
                    if base is not None:
                        self.builder.run_solvate(
                            base, custom_command=self.custom_cmds.get("solvate")
                        )
                finally:
                    os.chdir(original_cwd)

        self._log_success(
            "FEP solvate workflow completed for A_to_B and B_to_A directories."
        )

    def _determine_genions_base(self):
        """
        Determine the base name for genions based on available files.
        Returns the base name for genions or None if no suitable files found.
        """
        if os.path.isfile("complex.gro"):
            return "complex"
        elif not os.path.isfile("protein.gro"):
            # Ligand-only: look for ligandX.gro or hybrid_stateX.gro
            # Check for hybrid files first (FEP case)
            hybrid_gro_files = ["hybrid_stateA.gro", "hybrid_stateB.gro"]
            for hybrid_file in hybrid_gro_files:
                if os.path.isfile(hybrid_file):
                    return hybrid_file[:-4]  # strip .gro

            # Check for regular ligand files
            ligand_gro_files = [f"ligand{c}.gro" for c in string.ascii_uppercase]
            for fname in ligand_gro_files:
                if os.path.isfile(fname):
                    return fname[:-4]  # strip .gro

            return None
        else:
            return "protein"

    def do_genions(self, arg):
        """
        Run genions with optional custom command override. Handles three cases:
        1) Protein-only: adds ions to protein.solv.gro
        2) Protein + single ligand: adds ions to complex.solv.gro
        3) Ligand-only: adds ions to ligandX.solv.gro in current directory
        4) FEP directories: adds ions to hybrid_stateX.solv.gro or complex.solv.gro in each A_to_B/B_to_A directory
        Usage: "genions"
        Other Options: use "set genions" to override defaults
        """
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("genions_command")

        def run_genions_and_capture(basename, custom_command=None):
            # The base class _run_gromacs_command_internal already handles the "No default Per. Imp. Dih. types" error
            # automatically by commenting out the problematic line and rerunning. No need for redundant error handling here.
            try:
                self.builder.run_genions(basename, custom_command=custom_command)
                return True, ""
            except Exception as e:
                return False, str(e)

        try:
            # Check for FEP directories first
            fep_check_op = self.runtime_monitor.start_operation("check_fep_directories_genions")
            fep_dirs = [d for d in ["ligand_only", "protein_complex"] if os.path.isdir(d)]
            self.runtime_monitor.end_operation(success=True)

            if fep_dirs:
                fep_op = self.runtime_monitor.start_operation("genions_fep_workflow")
                self._do_genions_fep(fep_dirs)
                self.runtime_monitor.end_operation(success=True)
                return

            # Regular workflow
            base_check_op = self.runtime_monitor.start_operation("determine_genions_base")
            solvated_base = self._determine_genions_base()
            self.runtime_monitor.end_operation(success=True)

            if solvated_base is None:
                self._log_error("No suitable files found for genions.")
                self.runtime_monitor.end_operation(success=False, error_message="No suitable files found for genions")
                return

            pdb_check_op = self.runtime_monitor.start_operation("require_pdb_check_genions")
            if not self._require_pdb():
                self.runtime_monitor.end_operation(success=False, error_message="PDB requirement not met")
                return
            self.runtime_monitor.end_operation(success=True)

            # Run genions - the base class handles "No default Per. Imp. Dih. types" errors automatically
            genions_op = self.runtime_monitor.start_operation("run_genions")
            success, error_message = run_genions_and_capture(
                solvated_base, custom_command=self.custom_cmds.get("genions")
            )
            if success:
                self._log_success(f"Added ions to {solvated_base}.solv.gro")
                self.runtime_monitor.end_operation(success=True)
            else:
                self._log_error(f"Failed to add ions: {error_message}")
                self.runtime_monitor.end_operation(success=False, error_message=error_message)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def _do_genions_fep(self, fep_dirs):
        """
        Run genions for FEP workflow (ligand_only and protein_complex directories at A_to_B and B_to_A level).
        """
        self._log_info("Detected FEP directories: {}".format(", ".join(fep_dirs)))

        def run_genions_and_capture(basename, custom_command=None):
            # The base class _run_gromacs_command_internal already handles the "No default Per. Imp. Dih. types" error
            # automatically by commenting out the problematic line and rerunning. No need for redundant error handling here.
            try:
                self.builder.run_genions(basename, custom_command=custom_command)
                return True, ""
            except Exception as e:
                return False, str(e)

        # Process ligand_only directories
        ligand_only_dir = "ligand_only"
        if os.path.isdir(ligand_only_dir):
            self._log_info(f"Processing {ligand_only_dir} directories...")

            # Process A_to_B directory
            a_to_b_dir = os.path.join(ligand_only_dir, "A_to_B")
            if os.path.isdir(a_to_b_dir):
                self._log_info(f"Processing {a_to_b_dir}...")
                # Change to A_to_B directory and run genions
                original_cwd = os.getcwd()
                try:
                    os.chdir(a_to_b_dir)
                    base = "hybrid_stateA"  # Use hybrid_stateA for A_to_B
                    if os.path.isfile(f"{base}.solv.gro"):
                        success, error_message = run_genions_and_capture(
                            base, custom_command=self.custom_cmds.get("genions")
                        )
                        if success:
                            self._log_success(
                                f"Added ions to {base}.solv.gro in {a_to_b_dir}"
                            )

                        else:
                            self._log_error(
                                f"Failed to add ions in {a_to_b_dir}: {error_message}"
                            )
                    else:
                        self._log_error(f"{base}.solv.gro not found in {a_to_b_dir}")
                finally:
                    os.chdir(original_cwd)

            # Process B_to_A directory
            b_to_a_dir = os.path.join(ligand_only_dir, "B_to_A")
            if os.path.isdir(b_to_a_dir):
                self._log_info(f"Processing {b_to_a_dir}...")
                # Change to B_to_A directory and run genions
                original_cwd = os.getcwd()
                try:
                    os.chdir(b_to_a_dir)
                    base = "hybrid_stateB"  # Use hybrid_stateB for B_to_A
                    if os.path.isfile(f"{base}.solv.gro"):
                        success, error_message = run_genions_and_capture(
                            base, custom_command=self.custom_cmds.get("genions")
                        )
                        if success:
                            self._log_success(
                                f"Added ions to {base}.solv.gro in {b_to_a_dir}"
                            )

                        else:
                            self._log_error(
                                f"Failed to add ions in {b_to_a_dir}: {error_message}"
                            )
                    else:
                        self._log_error(f"{base}.solv.gro not found in {b_to_a_dir}")
                finally:
                    os.chdir(original_cwd)

        # Process protein_complex directories
        protein_complex_dir = "protein_complex"
        if os.path.isdir(protein_complex_dir):
            self._log_info(f"Processing {protein_complex_dir} directories...")

            # Process A_to_B directory
            a_to_b_dir = os.path.join(protein_complex_dir, "A_to_B")
            if os.path.isdir(a_to_b_dir):
                self._log_info(f"Processing {a_to_b_dir}...")
                # Change to A_to_B directory and run genions
                original_cwd = os.getcwd()
                try:
                    os.chdir(a_to_b_dir)
                    base = "complex"  # Use complex for protein_complex
                    if os.path.isfile(f"{base}.solv.gro"):
                        success, error_message = run_genions_and_capture(
                            base, custom_command=self.custom_cmds.get("genions")
                        )
                        if success:
                            self._log_success(
                                f"Added ions to {base}.solv.gro in {a_to_b_dir}"
                            )

                        else:
                            self._log_error(
                                f"Failed to add ions in {a_to_b_dir}: {error_message}"
                            )
                    else:
                        self._log_error(f"{base}.solv.gro not found in {a_to_b_dir}")
                finally:
                    os.chdir(original_cwd)

            # Process B_to_A directory
            b_to_a_dir = os.path.join(protein_complex_dir, "B_to_A")
            if os.path.isdir(b_to_a_dir):
                self._log_info(f"Processing {b_to_a_dir}...")
                # Change to B_to_A directory and run genions
                original_cwd = os.getcwd()
                try:
                    os.chdir(b_to_a_dir)
                    base = "complex"  # Use complex for protein_complex
                    if os.path.isfile(f"{base}.solv.gro"):
                        success, error_message = run_genions_and_capture(
                            base, custom_command=self.custom_cmds.get("genions")
                        )
                        if success:
                            self._log_success(
                                f"Added ions to {base}.solv.gro in {b_to_a_dir}"
                            )

                        else:
                            self._log_error(
                                f"Failed to add ions in {b_to_a_dir}: {error_message}"
                            )
                    else:
                        self._log_error(f"{base}.solv.gro not found in {b_to_a_dir}")
                finally:
                    os.chdir(original_cwd)

        self._log_success(
            "FEP genions workflow completed for A_to_B and B_to_A directories."
        )

    def do_em(self, arg):
        """Run energy minimization."""
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("em_command")

        try:
            pdb_check_op = self.runtime_monitor.start_operation("require_pdb_check_em")
            if not self._require_pdb():
                self.runtime_monitor.end_operation(success=False, error_message="PDB requirement not met")
                return
            self.runtime_monitor.end_operation(success=True)

            em_op = self.runtime_monitor.start_operation("run_energy_minimization")
            self.gmx.run_em(self.basename, arg)
            self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def do_nvt(self, arg):
        """Run NVT equilibration."""
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("nvt_command")

        try:
            pdb_check_op = self.runtime_monitor.start_operation("require_pdb_check_nvt")
            if not self._require_pdb():
                self.runtime_monitor.end_operation(success=False, error_message="PDB requirement not met")
                return
            self.runtime_monitor.end_operation(success=True)

            nvt_op = self.runtime_monitor.start_operation("run_nvt_equilibration")
            self.gmx.run_nvt(self.basename, arg)
            self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def do_npt(self, arg):
        """Run NPT equilibration."""
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("npt_command")

        try:
            pdb_check_op = self.runtime_monitor.start_operation("require_pdb_check_npt")
            if not self._require_pdb():
                self.runtime_monitor.end_operation(success=False, error_message="PDB requirement not met")
                return
            self.runtime_monitor.end_operation(success=True)

            npt_op = self.runtime_monitor.start_operation("run_npt_equilibration")
            self.gmx.run_npt(self.basename, arg)
            self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def do_production(self, arg):
        """Run production MD simulation."""
        # Start operation monitoring
        op_metrics = self.runtime_monitor.start_operation("production_command")

        try:
            pdb_check_op = self.runtime_monitor.start_operation("require_pdb_check_production")
            if not self._require_pdb():
                self.runtime_monitor.end_operation(success=False, error_message="PDB requirement not met")
                return
            self.runtime_monitor.end_operation(success=True)

            production_op = self.runtime_monitor.start_operation("run_production_simulation")
            self.gmx.run_production(self.basename, arg)
            self.runtime_monitor.end_operation(success=True)

        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise
        else:
            self.runtime_monitor.end_operation(success=True)

    def complete_tremd_prep(self, text, line, begidx, endidx):
        """Tab completion for tremd_prep command."""
        return self._complete_filename(text, ".gro", line, begidx, endidx)

    def do_tremd_prep(self, arg):
        """
        Calculate temperature ladder for TREMD simulations.

        Usage:
            tremd_prep                    # Interactive mode - prompts for parameters
            tremd_prep -i 300 -f 400 -p 0.2  # Command-line mode with parameters

        Interactive Mode:
            When no arguments are provided, the command will prompt you for:
            - Initial temperature (K): Lowest temperature in the ladder
            - Final temperature (K): Highest temperature in the ladder
            - Exchange probability: Desired probability between adjacent replicas (e.g., 0.2)

        Command-line Mode:
            -i INITIAL    Initial temperature (K)
            -f FINAL      Final temperature (K)
            -p PROB       Exchange probability (0-1)
        """
        # Parse arguments if provided
        args = self._parse_tremd_prep(arg) if arg.strip() else None

        solvated_gro = (
            "complex.solv.ions.gro"
            if os.path.exists("complex.solv.ions.gro")
            else "protein.solv.ions.gro"
        )
        if not os.path.isfile(solvated_gro):
            self._log_error(f"File not found: {solvated_gro}")
            return

        python_exe = sys.executable
        tremd_prep_path = os.path.join(os.path.dirname(__file__), "tremd_prep.py")

        if args and args.i is not None and args.f is not None and args.p is not None:
            # Use command-line arguments if provided
            command_str = f'"{python_exe}" "{tremd_prep_path}" "{solvated_gro}" -i {args.i} -f {args.f} -p {args.p}'
            self._log_info(f"Running with command-line arguments: {command_str}")
            success = self._execute_command(
                command=command_str,
                description="TREMD temperature ladder calculation (non-interactive)",
            )
            if not success:
                self._log_error("TREMD temperature ladder calculation failed.")
        else:
            # Use interactive mode
            self._log_info("Running TREMD temperature ladder calculation in interactive mode...")
            success = self._run_tremd_prep_interactive(tremd_prep_path, solvated_gro)
            if not success:
                self._log_error("TREMD temperature ladder calculation failed.")

    def _run_tremd_prep_interactive(self, tremd_prep_path: str, gro_file: str) -> bool:
        """
        Run tremd_prep in interactive mode, handling user input properly.

        Args:
            tremd_prep_path: Path to the tremd_prep.py script
            gro_file: Path to the GRO file to analyze

        Returns:
            bool: True if successful, False otherwise
        """
        python_exe = sys.executable
        cmd = [python_exe, tremd_prep_path, gro_file]

        self._log_info(f"Starting interactive TREMD prep: {' '.join(cmd)}")

        try:
            # Use a simpler approach - run the script with pre-defined input
            # This avoids the complex I/O handling issues

            # Get user input first
            print("TREMD Temperature Ladder Generator")
            print("=" * 40)

            try:
                initial_temp = input("Enter initial temperature (K): ")
                final_temp = input("Enter final temperature (K): ")
                exchange_prob = input("Enter desired exchange probability (e.g. 0.2): ")
            except KeyboardInterrupt:
                self._log_info("TREMD prep interrupted by user.")
                return False

            # Validate input
            try:
                initial_temp = float(initial_temp)
                final_temp = float(final_temp)
                exchange_prob = float(exchange_prob)
            except ValueError:
                self._log_error("Invalid input: temperatures and probability must be numbers.")
                return False

            # Prepare input data for the script
            input_data = f"{initial_temp}\n{final_temp}\n{exchange_prob}\n"

            # Run the script with the input data
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            # Display the output
            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(f"Errors: {result.stderr}")

            if result.returncode == 0:
                self._log_success("TREMD temperature ladder calculation completed successfully.")
                return True
            else:
                self._log_error(f"TREMD prep failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            self._log_error("TREMD prep timed out after 60 seconds.")
            return False
        except Exception as e:
            self._log_error(f"Error running TREMD prep: {e}")
            return False

    def _parse_tremd_prep(self, arg):

        parser = argparse.ArgumentParser(
            description="Calculate Replicas Temperature Replica Exchange"
        )
        parser.add_argument(
            "-i", type=float, help="Initial Temperature (K)"
        )
        parser.add_argument(
            "-f", type=float, help="Final Temperature (K)"
        )
        parser.add_argument(
            "-p", type=float, help="Probability of Exchange (0-1)"
        )
        # Use shlex.split to properly handle negative values and quoted strings
        return parser.parse_args(shlex.split(arg))

    def complete_demux(self, text, line, begidx, endidx):
        """
        Tab completion for demux command.

        Provides completion for directory names that might contain replica data.

        Args:
            text: The current input text to match
            line: The complete command line
            begidx: Beginning index of the word being completed
            endidx: Ending index of the word being completed

        Returns:
            List of matching directory names
        """
        # Look for directories in current directory
        directories = [d for d in os.listdir() if os.path.isdir(d)]

        # Filter based on current input
        if not text:
            return directories
        return [d for d in directories if d.startswith(text)]

    def do_demux(self, arg):
        """
        Run demultiplexing workflow for replica exchange simulations.

        This command processes replica exchange molecular dynamics (REMD) data
        by demultiplexing trajectories and generating analysis files.

        The workflow consists of several steps:
        1. Detect replica directories (numbered subdirectories)
        2. Aggregate log files from all replicas
        3. Run demux script to generate index files
        4. Demultiplex trajectories using the index files

        Usage:
            demux <input_directory>    # Process replicas in specified directory

        Required Files:
            - <input_directory>/<replica_number>/md.log: Log files from each replica
            - <input_directory>/<replica_number>/md.xtc: Trajectory files from each replica
            - demux.pl: Demux script (must be in PATH)

        Output Files:
            - <input_directory>/log_tmp/REMD.log: Combined log file
            - <input_directory>/log_tmp/replica_index.xvg: Replica index file
            - <input_directory>/log_tmp/replica_temp.xvg: Temperature index file
            - demuxed trajectory files in current directory

        Examples:
            demux remd_simulation    # Process replicas in remd_simulation directory
            demux .                  # Process replicas in current directory

        Note:
            This command requires the demux.pl script to be available in the system PATH.
            The script is typically provided with GROMACS REMD installations.
        """
        if not arg.strip():
            self._log_error("Usage: demux <input_directory>")
            return

        input_dir = arg.strip()

        # Check if input directory exists
        if not os.path.isdir(input_dir):
            self._log_error(f"Input directory not found: {input_dir}")
            return

        # Check if demux script is available
        demux_script = "demux.pl"
        if not shutil.which(demux_script):
            self._log_error(f"Demux script not found in PATH: {demux_script}")
            self._log_info("Please ensure demux.pl is installed and available in your PATH")
            return

        self.gmx.run_demux(input_dir, arg)

    def complete_autoimage(self, text, line, begidx, endidx):
        """
        Tab completion for autoimage command.

        Provides completion for common basename patterns based on existing .tpr files.

        Args:
            text: The current input text to match
            line: The complete command line
            begidx: Beginning index of the word being completed
            endidx: Ending index of the word being completed

        Returns:
            List of matching basenames from existing .tpr files
        """
        # Look for .tpr files in current directory
        tpr_files = [f for f in os.listdir() if f.endswith('.tpr')]

        # Extract basenames (remove .tpr extension)
        basenames = [os.path.splitext(f)[0] for f in tpr_files]

        # Filter based on current input
        if not text:
            return basenames
        return [b for b in basenames if b.startswith(text)]

    def do_autoimage(self, arg):
        """
        Run autoimage workflow to process trajectory files.

        This command executes a series of GROMACS trjconv commands to properly
        image and center trajectory files, creating a clean PDB representation.

        The workflow consists of three steps:
        1. Apply periodic boundary conditions (whole molecules)
        2. Center the system
        3. Create a PDB file with proper imaging

        Usage:
            autoimage              # Use default 'production' basename
            autoimage <basename>   # Use specified basename

        Required Files:
            - <basename>.tpr: GROMACS topology file
            - <basename>.xtc: GROMACS trajectory file

        Output Files:
            - <basename>.pbc1.xtc: Trajectory with PBC applied
            - <basename>.noPBC.xtc: Centered trajectory
            - <basename>.pdb: Final PDB file with proper imaging

        Examples:
            autoimage              # Process production.tpr/xtc files
            autoimage npt          # Process npt.tpr/xtc files
        """
        # Parse basename from arguments, default to 'production'
        basename = arg.strip() if arg.strip() else "production"

        # Check if required files exist
        tpr_file = f"{basename}.tpr"
        xtc_file = f"{basename}.xtc"

        if not os.path.isfile(tpr_file):
            self._log_error(f"Required file not found: {tpr_file}")
            return

        if not os.path.isfile(xtc_file):
            self._log_error(f"Required file not found: {xtc_file}")
            return

        self.gmx.run_autoimage(basename, arg)

    def complete_autoimage(self, text, line, begidx, endidx):
        """
        Tab completion for autoimage command.

        Provides completion for common basename patterns based on existing .tpr files.

        Args:
            text: The current input text to match
            line: The complete command line
            begidx: Beginning index of the word being completed
            endidx: Ending index of the word being completed

        Returns:
            List of matching basenames from existing .tpr files
        """
        # Look for .tpr files in current directory
        tpr_files = [f for f in os.listdir() if f.endswith('.tpr')]

        # Extract basenames (remove .tpr extension)
        basenames = [os.path.splitext(f)[0] for f in tpr_files]

        # Filter based on current input
        if not text:
            return basenames
        return [b for b in basenames if b.startswith(text)]

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
            self._log_info(
                "Use 'slurm' command to generate scripts with sourced ITP files."
            )

    def do_slurm(self, arg):
        """Generate SLURM scripts for MD, TREMD, or FEP simulations."""

        args = arg.strip().split()
        if len(args) < 2:
            self._log_error("Usage: slurm <sim_type> <hardware> [basename]")
            self._log_info("sim_type: 'md', 'tremd', or 'fep'")
            self._log_info("hardware: 'cpu' or 'gpu'")
            return

        sim_type = args[0].lower()
        hardware = args[1].lower()
        basename = args[2] if len(args) > 2 else self.basename

        if sim_type not in ["md", "tremd", "fep"]:
            self._log_error("sim_type must be 'md', 'tremd', or 'fep'")
            return

        if hardware not in ["cpu", "gpu"]:
            self._log_error("hardware must be 'cpu' or 'gpu'")
            return

        slurm_writer = SlurmWriter(logger=self.logger, debug=self.debug)
        slurm_writer.write_slurm_scripts(
            sim_type, hardware, basename, self.ligand_pdb_path
        )

    def print_random_quote(self):
        """Print a random quote from the quotes file."""
        try:
            # Get the path to the assets directory relative to this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(os.path.dirname(module_dir), "assets")
            quotes_path = os.path.join(assets_dir, "quotes.txt")

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
        self._log_info("YAGWIP reminds you:")
        self.print_random_quote()
        self._log_info("Goodbye!")
        return True


def main() -> None:
    """
    Main entry point for YAGWIP command-line interface.

    This function serves as the primary entry point for the YAGWIP application,
    handling command-line argument parsing and routing to appropriate execution
    modes (interactive, file-based, or batch processing).

    The function supports several execution modes:

    1. Interactive Mode (default):
       - Launches the interactive YAGWIP shell
       - Provides command-line interface with tab completion
       - Suitable for interactive molecular dynamics workflow setup

    2. File-based Mode:
       - Executes commands from a script file
       - Useful for automated workflows and reproducibility
       - Commands are executed sequentially from the file

    3. Batch Processing Mode:
       - Processes multiple PDB files using the same command script
       - Supports both sequential and parallel execution
       - Generates comprehensive reports and logs

    Command-line Arguments:
        -i, --interactive: Run interactive CLI (default if no other mode specified)
        -f, --file: Execute commands from input file
        -b, --batch: Batch process multiple PDBs using command script
        -p, --pdb-list: File containing list of PDB paths for batch processing
        -d, --pdb-dir: Directory containing PDB files for batch processing
        -r, --resume: Resume previous batch run
        --ligand_builder: Use ligand builder for batch processing
        --parallel: Enable parallel batch processing
        --workers: Number of parallel workers (default: auto-detect)
        --gmx-path: GROMACS executable path (default: "gmx")
        --debug: Enable debug mode

    Examples:
        # Interactive mode
        yagwip

        # Execute commands from file
        yagwip -f workflow.txt

        # Batch process with ligand builder
        yagwip -b commands.txt -p pdb_list.txt --ligand_builder

        # Parallel batch processing
        yagwip -b commands.txt -d pdb_directory/ --parallel --workers 4

        # Resume interrupted batch run
        yagwip -b commands.txt -p pdb_list.txt --resume

    Exit Codes:
        0: Successful execution
        1: Error during execution (file not found, command failed, etc.)

    Note:
        The function automatically validates GROMACS installation and provides
        helpful error messages if dependencies are missing.
    """
    parser = argparse.ArgumentParser(
        description="YAGWIP - Yet Another GROMACS Wrapper In Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yagwip                           # Interactive mode
  yagwip -f workflow.txt           # Execute commands from file
  yagwip -b commands.txt -p pdb_list.txt --ligand_builder  # Batch processing
  yagwip -b commands.txt -d pdb_directory/ --parallel      # Parallel batch processing
        """
    )

    # Execution mode arguments
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Run interactive CLI (default if no other mode specified)"
    )
    parser.add_argument(
        "-f", "--file", type=str,
        help="Execute commands from input file"
    )

    # Batch processing arguments
    parser.add_argument(
        "-b", "--batch", type=str,
        help="Batch process multiple PDBs using command script"
    )
    parser.add_argument(
        "-p", "--pdb-list", type=str,
        help="File containing list of PDB paths for batch processing"
    )
    parser.add_argument(
        "-d", "--pdb-dir", type=str,
        help="Directory containing PDB files for batch processing"
    )
    parser.add_argument(
        "-r", "--resume", action="store_true",
        help="Resume previous batch run"
    )
    parser.add_argument(
        "--ligand_builder", action="store_true",
        help="Use ligand builder for batch processing"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Enable parallel batch processing"
    )
    parser.add_argument(
        "--workers", type=int,
        help="Number of parallel workers (default: auto-detect)"
    )

    # Configuration arguments
    parser.add_argument(
        "--gmx-path", type=str, default=None,
        help="GROMACS executable path (default: auto-detect gmx_mpi or gmx)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Initialize YAGWIP shell
    cli = YagwipShell(args.gmx_path)

    if args.debug:
        cli.do_debug("on")

    # Handle batch processing
    if args.batch:
        # Determine number of workers
        max_workers = args.workers
        if args.parallel and not max_workers:
            max_workers = min(mp.cpu_count(), 8)  # Auto-detect with cap

        # Initialize batch processor
        if args.parallel:
            print(f"Initializing parallel batch processor with {max_workers} workers...")
            batch_processor = ParallelBatchProcessor(
                gmx_path=args.gmx_path,
                debug=args.debug,
                logger=cli.logger,
                ligand_builder=args.ligand_builder,
                max_workers=max_workers,
            )
        else:
            print("Initializing sequential batch processor...")
            batch_processor = ParallelBatchProcessor(
                gmx_path=args.gmx_path,
                debug=args.debug,
                logger=cli.logger,
                ligand_builder=args.ligand_builder,
                max_workers=1,  # Sequential processing
            )

        # Load PDB files
        if args.pdb_list:
            # Load from PDB list file
            batch_processor.load_pdb_list(args.pdb_list)
        elif args.pdb_dir:
            # Load from directory
            batch_processor.load_pdb_directory(args.pdb_dir)
        else:
            print("[ERROR] Must specify either --pdb-list or --pdb-dir for batch processing")
            sys.exit(1)

        # Execute batch
        print(f"Starting batch processing with {len(batch_processor.jobs)} jobs...")
        if args.ligand_builder:
            print("Ligand builder enabled for batch processing")
        if args.parallel:
            print(f"Parallel processing enabled with {max_workers} workers")

        results = batch_processor.execute_batch(args.batch, resume=args.resume)

        if results:
            print(f"Batch processing completed. Results saved in {batch_processor.results_dir}")
            print(f"Completed: {results['completed_jobs']}/{results['total_jobs']} jobs")
            print(f"Failed: {results['failed_jobs']} jobs")
            if args.parallel:
                print(f"Parallel workers used: {results.get('parallel_workers', 'N/A')}")
        else:
            print("Batch processing failed.")
            sys.exit(1)

    # Handle single file processing (original functionality)
    elif args.file:
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

    # Interactive mode (default)
    else:
        cli.cmdloop()


if __name__ == "__main__":
    main()
