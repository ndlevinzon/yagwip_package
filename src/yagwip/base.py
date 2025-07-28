"""
base.py -- Base classes and common patterns for YAGWIP
"""

# === Standard Library Imports ===
import os
import shutil
from abc import ABC
from typing import Optional, Dict, Any, List
from enum import Enum

# === Local Imports ===
from utils.log_utils import (
    LoggingMixin,
    setup_logger,
    auto_monitor,
    runtime_context,
    command_context,
)


class LogLevel(Enum):
    """Standardized log levels for consistent messaging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class YagwipBase(LoggingMixin, ABC):
    """
    Base class for all YAGWIP components providing common functionality.

    This class standardizes:
    - Initialization patterns
    - Logging setup
    - Debug mode handling
    - File operations
    - Error handling
    - Configuration management
    - Automatic runtime monitoring
    """

    def __init__(
        self, gmx_path: Optional[str] = None, debug: bool = False, logger=None, **kwargs
    ):
        """
        Initialize base YAGWIP component.

        Args:
            gmx_path: Path to GROMACS executable
            debug: Enable debug mode
            logger: Logger instance (if None, creates one)
            **kwargs: Additional configuration
        """
        super().__init__()
        self.gmx_path = gmx_path
        self.debug = debug
        self.logger = logger or setup_logger(debug_mode=debug)

        # Store additional configuration
        self.config = kwargs

        # Initialize component-specific setup
        self._setup()

    def _setup(self):
        """Component-specific initialization. Override in subclasses."""
        pass

    @auto_monitor
    def _validate_file_exists(self, filepath: str, description: str = "File") -> bool:
        """
        Validate that a file exists and log appropriate messages.

        Args:
            filepath: Path to check
            description: Description for logging

        Returns:
            True if file exists, False otherwise
        """
        if not os.path.exists(filepath):
            self._log_error(f"{description} not found: {filepath}")
            return False
        self._log_debug(f"{description} validated: {filepath}")
        return True

    @auto_monitor
    def _validate_directory_exists(
        self, dirpath: str, description: str = "Directory"
    ) -> bool:
        """
        Validate that a directory exists and create if needed.

        Args:
            dirpath: Path to check
            description: Description for logging

        Returns:
            True if directory exists or was created, False otherwise
        """
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath, exist_ok=True)
                self._log_info(f"Created {description}: {dirpath}")
            except Exception as e:
                self._log_error(f"Failed to create {description}: {e}")
                return False
        return True

    @auto_monitor
    def _safe_copy_file(self, src: str, dst: str, description: str = "File") -> bool:
        """
        Safely copy a file with error handling and logging.

        Args:
            src: Source file path
            dst: Destination file path
            description: Description for logging

        Returns:
            True if copy successful, False otherwise
        """
        try:
            if not self._validate_file_exists(src, description):
                return False

            shutil.copy2(src, dst)
            self._log_success(f"Copied {description}: {src} -> {dst}")
            return True
        except Exception as e:
            self._log_error(f"Failed to copy {description}: {e}")
            return False

    @auto_monitor
    def _execute_command(
        self,
        command: str,
        description: str = "Command",
        pipe_input: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Execute a command with standardized error handling and logging.

        Args:
            command: Command to execute
            description: Description for logging
            pipe_input: Input to pipe to command
            **kwargs: Additional arguments for run_gromacs_command

        Returns:
            True if command successful, False otherwise
        """

        # Enhanced debug mode: Show detailed information but still execute
        if self.debug:
            self._log_debug("=" * 60)
            self._log_debug(f"DEBUG MODE: {description}")
            self._log_debug("=" * 60)
            self._log_debug(f"Command: {command}")
            if pipe_input:
                self._log_debug(f"Pipe Input: {pipe_input}")
            self._log_debug(f"Additional Args: {kwargs}")

            # Show current system resources
            try:
                import psutil

                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                disk = psutil.disk_usage(".")

                self._log_debug("System Resources:")
                self._log_debug(f"  CPU Usage: {cpu_percent:.1f}%")
                self._log_debug(
                    f"  Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.2f} GB used)"
                )
                self._log_debug(f"  Disk Usage: {disk.percent:.1f}%")
                self._log_debug(
                    f"  Available Memory: {memory.available / (1024**3):.2f} GB"
                )
            except ImportError:
                self._log_debug(
                    "System monitoring not available (psutil not installed)"
                )
            except Exception as e:
                self._log_debug(f"Could not get system info: {e}")

            self._log_debug("=" * 60)
            self._log_debug("EXECUTING COMMAND (Debug mode still runs commands)")
            self._log_debug("=" * 60)

        # Use runtime monitoring for command execution
        return self._log_with_runtime(
            f"Execute {description}",
            self._run_gromacs_command_internal,
            command,
            pipe_input,
            **kwargs,
        )

    def _run_gromacs_command_internal(
        self, command: str, pipe_input: Optional[str] = None
    ) -> bool:
        """Internal method to run GROMACS command with runtime monitoring."""
        import subprocess
        import re

        # Log the command about to be executed
        self._log_info(f"[RUNNING] {command}")

        # In debug mode, skip execution and only log/print a message
        if self.debug:
            msg = "[DEBUG MODE] Command not executed."
            self._log_debug(msg)
            return True  # Or False, depending on your policy

        try:
            # Execute the command with optional piped input
            result = subprocess.run(
                command,
                input=pipe_input,  # Piped input to stdin, if any (e.g., group number)
                shell=True,  # Run command through shell
                capture_output=True,  # Capture both stdout and stderr
                text=True,  # Decode outputs as strings instead of bytes
                check=False,  # Do not raise an error if the command fails
            )

            # Strip leading/trailing whitespace from stderr and stdout
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            error_text = (
                f"{stderr}\n{stdout}".lower()
            )  # Combined output for error checks

            # Check if the command failed based on return code
            if result.returncode != 0:
                err_msg = f"[!] Command failed with return code {result.returncode}"
                self._log_error(err_msg)
                if stderr:
                    self._log_error(f"[STDERR] {stderr}")
                if stdout:
                    self._log_info(f"[STDOUT] {stdout}")
                print(err_msg)
                if stderr:
                    print("[STDERR]", stderr)
                if stdout:
                    print("[STDOUT]", stdout)

                # Catch atom number mismatch error
                if "number of coordinates in coordinate file" in error_text:
                    specific_msg = "[!] Check ligand and protonation: .gro and .top files have different atom counts."
                    self._log_warning(specific_msg)
                    print(specific_msg)

                # Catch periodic improper dihedral type error
                elif "No default Per. Imp. Dih. types" in error_text:
                    match = re.search(
                        r"\[file topol\.top, line \d+\]", stderr, re.IGNORECASE
                    )
                    if match:
                        line_num = int(match.group(1))
                        top_path = "./topol.top"
                        try:
                            with open(top_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                            if 0 <= line_num - 1 < len(lines):
                                if not lines[line_num - 1].strip().startswith(";"):
                                    lines[line_num - 1] = f";{lines[line_num - 1]}"
                                    with open(top_path, "w", encoding="utf-8") as f:
                                        f.writelines(lines)
                                    msg = (
                                        f"[#] Detected improper dihedral error, likely an artifact from AMBER forcefields."
                                        f" Commenting out line {line_num} in topol.top and rerunning..."
                                    )
                                    self._log_warning(msg)
                                    print(msg)
                                    # Retry the command
                                    retry_msg = "[#] Rerunning command after modifying topol.top..."
                                    self._log_info(retry_msg)
                                    print(retry_msg)
                                    # Recursive retry, but prevent infinite loops (user responsibility)
                                    return self._run_gromacs_command_internal(
                                        command,
                                        pipe_input,
                                    )
                        except Exception as e:
                            fail_msg = f"[!] Failed to modify topol.top: {e}"
                            self._log_error(fail_msg)
                            print(fail_msg)
                    else:
                        fallback_msg = "[!] Detected dihedral error, but couldn't find line number in topol.top."
                        self._log_warning(fallback_msg)
                        print(fallback_msg)

                return False  # Final return if not resolved

            else:
                # If successful, optionally print/log stdout
                if stdout:
                    self._log_info(f"[STDOUT] {stdout}")
                    print(stdout)
                return True

        except Exception as e:
            # Catch and log any unexpected runtime exceptions (e.g., permission issues)
            self._log_error(f"[!] Failed to run command: {e}")
            return False

    @auto_monitor
    def _get_file_basename(self, filepath: str) -> str:
        """Get basename without extension from filepath."""
        return os.path.splitext(os.path.basename(filepath))[0]

    @auto_monitor
    def _build_filepath(self, basename: str, extension: str) -> str:
        """Build filepath from basename and extension."""
        return f"{basename}.{extension}"

    @auto_monitor
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    @auto_monitor
    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        self._log_debug(f"Configuration updated: {key} = {value}")

    @auto_monitor
    def update_config(self, **kwargs):
        """Update multiple configuration values."""
        self.config.update(kwargs)
        self._log_debug(f"Configuration updated: {list(kwargs.keys())}")

    def monitor_operation(self, operation_name: str):
        """
        Context manager for monitoring operations with automatic resource tracking.

        Usage:
            with self.monitor_operation("my_operation"):
                # Code to monitor
                pass
        """
        return runtime_context(operation_name, self.logger, self.debug)

    def monitor_command(self, command_name: str):
        """
        Context manager for monitoring complete commands with summary logging.

        Usage:
            with self.monitor_command("loadpdb"):
                # All operations within this block will be tracked
                # Only summary will be logged unless debug=True
                pass
        """
        return command_context(command_name, self.logger, self.debug)

    def get_runtime_summary(self) -> Dict[str, Any]:
        """Get runtime summary for this component."""
        if hasattr(self, "runtime_monitor"):
            return self.runtime_monitor.get_summary()
        return {}


class ConfigurableMixin:
    """
    Mixin for classes that need configuration management.
    Provides methods for getting/setting configuration values.
    """

    def __init__(self, **kwargs):
        """Initialize with configuration."""
        self._config = kwargs

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value

    def update_config(self, **kwargs):
        """Update multiple configuration values."""
        self._config.update(kwargs)


class FileProcessorMixin:
    """Mixin for file processing operations with automatic monitoring."""

    @staticmethod
    def read_file_lines(filepath: str) -> List[str]:
        """Read file lines."""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.readlines()

    @staticmethod
    def write_file_lines(filepath: str, lines: List[str]):
        """Write file lines."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def append_to_file(filepath: str, content: str):
        """Append content to file."""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def backup_file(filepath: str, suffix: str = ".backup") -> str:
        """Backup file."""
        backup_path = filepath + suffix
        shutil.copy2(filepath, backup_path)
        return backup_path
