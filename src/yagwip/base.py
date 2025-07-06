"""
base.py -- Base classes and common patterns for YAGWIP
"""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from enum import Enum

from .log import LoggingMixin, setup_logger


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
    """

    def __init__(self,
                 gmx_path: Optional[str] = None,
                 debug: bool = False,
                 logger=None,
                 **kwargs):
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

    def _log_message(self, level: LogLevel, message: str, **kwargs):
        """
        Standardized logging method with consistent formatting.

        Args:
            level: Log level from LogLevel enum
            message: Message to log
            **kwargs: Additional context
        """
        formatted_message = f"[{level.value}] {message}"
        if kwargs:
            context = " ".join(f"{k}={v}" for k, v in kwargs.items())
            formatted_message += f" ({context})"

        self._log(formatted_message)

    def _log_debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.debug:
            self._log_message(LogLevel.DEBUG, message, **kwargs)

    def _log_info(self, message: str, **kwargs):
        """Log info message."""
        self._log_message(LogLevel.INFO, message, **kwargs)

    def _log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_message(LogLevel.WARNING, message, **kwargs)

    def _log_error(self, message: str, **kwargs):
        """Log error message."""
        self._log_message(LogLevel.ERROR, message, **kwargs)

    def _log_success(self, message: str, **kwargs):
        """Log success message."""
        self._log_message(LogLevel.SUCCESS, message, **kwargs)

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

    def _validate_directory_exists(self, dirpath: str, description: str = "Directory") -> bool:
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

    def _execute_command(self,
                         command: str,
                         description: str = "Command",
                         pipe_input: Optional[str] = None,
                         **kwargs) -> bool:
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
        from .utils import run_gromacs_command

        if self.debug:
            self._log_debug(f"{description}: {command}")
            return True

        # Use runtime monitoring for command execution
        return self._log_with_runtime(
            f"Execute {description}",
            self._run_gromacs_command_internal,
            command, pipe_input, **kwargs
        )

    def _run_gromacs_command_internal(self, command: str, pipe_input: Optional[str] = None, **kwargs) -> bool:
        """Internal method to run GROMACS command with runtime monitoring."""
        from .utils import run_gromacs_command

        self._log_info(f"Running command: {command}")
        success = run_gromacs_command(
            command=command,
            pipe_input=pipe_input,
            debug=self.debug,
            logger=self.logger,
            **kwargs
        )

        if success:
            self._log_success(f"Command completed successfully")
        else:
            self._log_error(f"Command failed")

        return success

    def _get_file_basename(self, filepath: str) -> str:
        """Get basename without extension from filepath."""
        return os.path.splitext(os.path.basename(filepath))[0]

    def _build_filepath(self, basename: str, extension: str) -> str:
        """Build filepath from basename and extension."""
        return f"{basename}.{extension}"

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        self._log_debug(f"Configuration updated: {key} = {value}")

    def update_config(self, **kwargs):
        """Update multiple configuration values."""
        self.config.update(kwargs)
        self._log_debug(f"Configuration updated: {list(kwargs.keys())}")


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
    """
    Mixin for classes that process files.
    Provides common file processing utilities.
    """

    @staticmethod
    def read_file_lines(filepath: str) -> List[str]:
        """Read file and return lines."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()

    @staticmethod
    def write_file_lines(filepath: str, lines: List[str]):
        """Write lines to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    @staticmethod
    def append_to_file(filepath: str, content: str):
        """Append content to file."""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def backup_file(filepath: str, suffix: str = ".backup") -> str:
        """Create backup of file and return backup path."""
        backup_path = filepath + suffix
        shutil.copy2(filepath, backup_path)
        return backup_path


class CommandExecutorMixin:
    """
    Mixin for classes that execute commands.
    Provides standardized command execution with logging.
    """

    def execute_command(self,
                        command: str,
                        description: str = "Command",
                        pipe_input: Optional[str] = None,
                        **kwargs) -> bool:
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
        from .utils import run_gromacs_command

        # Check if we have debug mode and logging capabilities
        debug_mode = getattr(self, 'debug', False)
        has_logging = hasattr(self, '_log_debug') and hasattr(self, '_log_info')

        if debug_mode:
            if has_logging:
                self._log_debug(f"{description}: {command}")
            else:
                print(f"[DEBUG] {description}: {command}")
            return True

        if has_logging:
            self._log_info(f"Running {description}")

        success = run_gromacs_command(
            command=command,
            pipe_input=pipe_input,
            debug=debug_mode,
            logger=getattr(self, 'logger', None),
            **kwargs
        )

        if has_logging and hasattr(self, '_log_success') and hasattr(self, '_log_error'):
            if success:
                self._log_success(f"{description} completed successfully")
            else:
                self._log_error(f"{description} failed")

        return success