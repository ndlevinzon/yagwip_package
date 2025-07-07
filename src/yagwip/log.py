import time
import logging
import psutil
import os
import platform
import functools
import threading
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager


class LogLevel(Enum):
    """Standardized log levels for consistent messaging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    RUNTIME = "RUNTIME"  # New level for runtime information


@dataclass
class SystemInfo:
    """System information snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    platform: str
    python_version: str
    process_id: int
    thread_id: Optional[int] = None


@dataclass
class RuntimeMetrics:
    """Runtime metrics for an operation."""

    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    system_info_start: Optional[SystemInfo] = None
    system_info_end: Optional[SystemInfo] = None
    memory_delta_mb: Optional[float] = None
    cpu_delta_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class CommandMetrics:
    """Metrics for a complete command execution."""

    command_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    system_info_start: Optional[SystemInfo] = None
    system_info_end: Optional[SystemInfo] = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    success: bool = True
    error_message: Optional[str] = None
    sub_operations: List[RuntimeMetrics] = field(default_factory=list)


class RuntimeMonitor:
    """Monitors runtime performance and system metrics."""

    def __init__(
        self, logger: Optional[logging.Logger] = None, debug_mode: bool = False
    ):
        self.logger = logger
        self.debug_mode = debug_mode
        self.metrics_history: List[RuntimeMetrics] = []
        self.current_operation: Optional[RuntimeMetrics] = None
        self.current_command: Optional[CommandMetrics] = None
        self._lock = threading.Lock()

    def _get_system_info(self) -> SystemInfo:
        """Capture current system information."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(".")
            return SystemInfo(
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                platform=f"{platform.system()} {platform.release()}",
                python_version=platform.python_version(),
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
            )
        except Exception as e:
            # Fallback if psutil is not available
            return SystemInfo(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                platform="Unknown",
                python_version=platform.python_version(),
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
            )

    def start_command(self, command_name: str) -> CommandMetrics:
        """Start monitoring a complete command."""
        with self._lock:
            self.current_command = CommandMetrics(
                command_name=command_name,
                start_time=datetime.now(),
                system_info_start=self._get_system_info(),
            )

            if self.logger and self.debug_mode:
                self.logger.info(f"Starting command: {command_name}")
                if self.current_command.system_info_start:
                    self._log_system_info(
                        "COMMAND_START", self.current_command.system_info_start
                    )

            return self.current_command

    def end_command(
        self, success: bool = True, error_message: Optional[str] = None
    ) -> Optional[CommandMetrics]:
        """End monitoring a command and log summary."""
        with self._lock:
            if not self.current_command:
                return None

            self.current_command.end_time = datetime.now()
            self.current_command.duration_seconds = (
                self.current_command.end_time - self.current_command.start_time
            ).total_seconds()
            self.current_command.system_info_end = self._get_system_info()
            self.current_command.success = success
            self.current_command.error_message = error_message

            # Calculate summary from sub-operations
            if self.current_command.sub_operations:
                self.current_command.total_operations = len(
                    self.current_command.sub_operations
                )
                self.current_command.successful_operations = sum(
                    1 for op in self.current_command.sub_operations if op.success
                )
                self.current_command.failed_operations = (
                    self.current_command.total_operations
                    - self.current_command.successful_operations
                )

            # Log command summary (always log, regardless of debug mode)
            if self.logger:
                self._log_command_summary(self.current_command)

            # Clear current command
            completed_command = self.current_command
            self.current_command = None

            return completed_command

    def start_operation(self, operation_name: str, **kwargs) -> RuntimeMetrics:
        """Start monitoring an operation."""
        with self._lock:
            self.current_operation = RuntimeMetrics(
                operation_name=operation_name,
                start_time=datetime.now(),
                system_info_start=self._get_system_info(),
                additional_info=kwargs,
            )

            # Only log detailed info in debug mode
            if self.logger and self.debug_mode:
                self.logger.info(f"Starting operation: {operation_name}")
                if self.current_operation.system_info_start:
                    self._log_system_info(
                        "START", self.current_operation.system_info_start
                    )

            return self.current_operation

    def end_operation(
        self, success: bool = True, error_message: Optional[str] = None
    ) -> Optional[RuntimeMetrics]:
        """End monitoring an operation and calculate metrics."""
        with self._lock:
            if not self.current_operation:
                return None

            self.current_operation.end_time = datetime.now()
            self.current_operation.duration_seconds = (
                self.current_operation.end_time - self.current_operation.start_time
            ).total_seconds()
            self.current_operation.system_info_end = self._get_system_info()
            self.current_operation.success = success
            self.current_operation.error_message = error_message

            # Calculate deltas
            if (
                self.current_operation.system_info_start
                and self.current_operation.system_info_end
            ):
                self.current_operation.memory_delta_mb = (
                    self.current_operation.system_info_end.memory_used_mb
                    - self.current_operation.system_info_start.memory_used_mb
                )
                self.current_operation.cpu_delta_percent = (
                    self.current_operation.system_info_end.cpu_percent
                    - self.current_operation.system_info_start.cpu_percent
                )

            # Only log detailed runtime information in debug mode
            if self.logger and self.debug_mode:
                self._log_runtime_summary(self.current_operation)

            # Add to current command if exists
            if self.current_command:
                self.current_command.sub_operations.append(self.current_operation)

            # Store in history
            self.metrics_history.append(self.current_operation)

            # Clear current operation
            completed_operation = self.current_operation
            self.current_operation = None

            return completed_operation

    def _log_system_info(self, prefix: str, system_info: SystemInfo):
        """Log system information."""
        if not self.logger:
            return

        self.logger.info(f"[{prefix}] System Info:")
        self.logger.info(f"  CPU Usage: {system_info.cpu_percent:.1f}%")
        self.logger.info(
            f"  Memory Usage: {system_info.memory_percent:.1f}% ({system_info.memory_used_mb:.1f} MB used, {system_info.memory_available_mb:.1f} MB available)"
        )
        self.logger.info(f"  Disk Usage: {system_info.disk_usage_percent:.1f}%")
        self.logger.info(f"  Platform: {system_info.platform}")
        self.logger.info(f"  Python: {system_info.python_version}")
        self.logger.info(f"  Process ID: {system_info.process_id}")
        if system_info.thread_id:
            self.logger.info(f"  Thread ID: {system_info.thread_id}")

    def _log_runtime_summary(self, metrics: RuntimeMetrics):
        """Log runtime summary for an operation."""
        if not self.logger:
            return

        status = "SUCCESS" if metrics.success else "FAILED"
        duration_str = (
            f"{metrics.duration_seconds:.2f}s"
            if metrics.duration_seconds
            else "unknown"
        )

        # Enhanced debug mode: Show detailed information
        if self.debug_mode:
            self.logger.info("=" * 50)
            self.logger.info(f"DEBUG: {metrics.operation_name} Operation Summary")
            self.logger.info("=" * 50)
            self.logger.info(f"Duration: {duration_str}")
            self.logger.info(
                f"Status: {'✓ SUCCESS' if metrics.success else '✗ FAILED'}"
            )

            if metrics.function_name:
                self.logger.info(f"Function: {metrics.function_name}")
            if metrics.module_name:
                self.logger.info(f"Module: {metrics.module_name}")
            if metrics.line_number:
                self.logger.info(f"Line: {metrics.line_number}")

            if metrics.system_info_start and metrics.system_info_end:
                self.logger.info("System Resource Changes:")
                self.logger.info(
                    f"  CPU: {metrics.system_info_start.cpu_percent:.1f}% → {metrics.system_info_end.cpu_percent:.1f}% (Δ: {metrics.cpu_delta_percent:.1f}%)"
                )
                self.logger.info(
                    f"  Memory: {metrics.system_info_start.memory_percent:.1f}% → {metrics.system_info_end.memory_percent:.1f}% (Δ: {metrics.memory_delta_mb:.2f} MB)"
                )
                self.logger.info(
                    f"  Available Memory: {metrics.system_info_start.memory_available_mb:.1f} MB → {metrics.system_info_end.memory_available_mb:.1f} MB"
                )

            if metrics.error_message:
                self.logger.error(f"Error Details: {metrics.error_message}")

            if metrics.additional_info:
                self.logger.info("Additional Information:")
                for key, value in metrics.additional_info.items():
                    self.logger.info(f"  {key}: {value}")

            self.logger.info("=" * 50)
        else:
            # Standard logging for non-debug mode
            self.logger.info(f"[RUNTIME] Operation: {metrics.operation_name}")
            self.logger.info(f"  Status: {status}")
            self.logger.info(f"  Duration: {duration_str}")

            if metrics.function_name:
                self.logger.info(f"  Function: {metrics.function_name}")
            if metrics.module_name:
                self.logger.info(f"  Module: {metrics.module_name}")
            if metrics.line_number:
                self.logger.info(f"  Line: {metrics.line_number}")

            if metrics.memory_delta_mb is not None:
                memory_change = "+" if metrics.memory_delta_mb >= 0 else ""
                self.logger.info(
                    f"  Memory Change: {memory_change}{metrics.memory_delta_mb:.1f} MB"
                )

            if metrics.cpu_delta_percent is not None:
                cpu_change = "+" if metrics.cpu_delta_percent >= 0 else ""
                self.logger.info(
                    f"  CPU Change: {cpu_change}{metrics.cpu_delta_percent:.1f}%"
                )

            if metrics.error_message:
                self.logger.error(f"  Error: {metrics.error_message}")

            # Log additional info if present
            if metrics.additional_info:
                for key, value in metrics.additional_info.items():
                    self.logger.info(f"  {key}: {value}")

    def _log_command_summary(self, metrics: CommandMetrics):
        """Log command summary."""
        if not self.logger:
            return

        status = "SUCCESS" if metrics.success else "FAILED"
        duration_str = (
            f"{metrics.duration_seconds:.2f}s"
            if metrics.duration_seconds
            else "unknown"
        )

        self.logger.info(f"[COMMAND_SUMMARY] Command: {metrics.command_name}")
        self.logger.info(f"  Status: {status}")
        self.logger.info(f"  Total Duration: {duration_str}")
        self.logger.info(f"  Total Operations: {metrics.total_operations}")
        self.logger.info(f"  Successful Operations: {metrics.successful_operations}")
        self.logger.info(f"  Failed Operations: {metrics.failed_operations}")

        if metrics.system_info_start and metrics.system_info_end:
            memory_delta = (
                metrics.system_info_end.memory_used_mb
                - metrics.system_info_start.memory_used_mb
            )
            cpu_delta = (
                metrics.system_info_end.cpu_percent
                - metrics.system_info_start.cpu_percent
            )

            memory_change = "+" if memory_delta >= 0 else ""
            cpu_change = "+" if cpu_delta >= 0 else ""

            self.logger.info(
                f"  Total Memory Change: {memory_change}{memory_delta:.1f} MB"
            )
            self.logger.info(f"  Total CPU Change: {cpu_change}{cpu_delta:.1f}%")

        if metrics.error_message:
            self.logger.error(f"  Error: {metrics.error_message}")

        # Log sub-operations if in debug mode
        if self.debug_mode and metrics.sub_operations:
            self.logger.info("  Sub-operations:")
            for op in metrics.sub_operations:
                duration = (
                    f"{op.duration_seconds:.2f}s" if op.duration_seconds else "unknown"
                )
                status = "SUCCESS" if op.success else "FAILED"
                self.logger.info(f"    - {op.operation_name}: {duration} ({status})")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all runtime metrics."""
        if not self.metrics_history:
            return {}

        total_operations = len(self.metrics_history)
        successful_operations = sum(1 for m in self.metrics_history if m.success)
        total_duration = sum(m.duration_seconds or 0 for m in self.metrics_history)
        avg_duration = total_duration / total_operations if total_operations > 0 else 0

        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": total_operations - successful_operations,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "success_rate": (
                successful_operations / total_operations if total_operations > 0 else 0
            ),
        }


class LoggingMixin:
    """Enhanced mixin class to provide consistent logging functionality with runtime monitoring."""

    def __init__(self):
        self.runtime_monitor = RuntimeMonitor(
            getattr(self, "logger", None), getattr(self, "debug", False)
        )

    def _log(self, msg):
        """Log message using logger or print if no logger available."""
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def _log_with_runtime(self, operation_name: str, func: Callable, *args, **kwargs):
        """Execute a function with runtime monitoring."""
        metrics = self.runtime_monitor.start_operation(operation_name)

        try:
            result = func(*args, **kwargs)
            self.runtime_monitor.end_operation(success=True)
            return result
        except Exception as e:
            self.runtime_monitor.end_operation(success=False, error_message=str(e))
            raise

    def runtime_monitored(self, operation_name: Optional[str] = None):
        """Decorator to monitor runtime of a method."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get operation name from function name if not provided
                op_name = operation_name or f"{func.__name__}"

                # Get the runtime monitor from the instance
                if hasattr(args[0], "runtime_monitor"):
                    monitor = args[0].runtime_monitor
                else:
                    # Fallback to basic timing
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        logger = getattr(args[0], "logger", None)
                        if logger:
                            logger.info(f"[RUNTIME] {op_name}: {duration:.2f}s")
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        logger = getattr(args[0], "logger", None)
                        if logger:
                            logger.error(
                                f"[RUNTIME] {op_name} FAILED: {duration:.2f}s - {str(e)}"
                            )
                        raise

                # Use the runtime monitor
                metrics = monitor.start_operation(op_name)
                try:
                    result = func(*args, **kwargs)
                    monitor.end_operation(success=True)
                    return result
                except Exception as e:
                    monitor.end_operation(success=False, error_message=str(e))
                    raise

            return wrapper

        return decorator


# Global runtime monitor for automatic monitoring
_global_runtime_monitor = RuntimeMonitor(logging.getLogger("yagwip"), debug_mode=False)


def auto_monitor(
    func: Optional[Callable] = None, *, operation_name: Optional[str] = None
):
    """
    Automatic runtime monitoring decorator that can be applied to any function.

    This decorator automatically tracks:
    - Execution time
    - Memory usage changes
    - CPU usage changes
    - Function metadata (name, module, line number)

    Usage:
        @auto_monitor
        def my_function():
            pass

        @auto_monitor(operation_name="Custom Name")
        def my_function():
            pass
    """

    def decorator(func_to_monitor):
        @wraps(func_to_monitor)
        def wrapper(*args, **kwargs):
            # Get function metadata
            import inspect

            frame = inspect.currentframe()
            if frame:
                try:
                    # Get caller information
                    caller_frame = frame.f_back
                    if caller_frame:
                        module_name = caller_frame.f_globals.get("__name__", "unknown")
                        line_number = caller_frame.f_lineno
                    else:
                        module_name = "unknown"
                        line_number = 0
                finally:
                    frame = None
            else:
                module_name = "unknown"
                line_number = 0

            # Determine operation name
            op_name = operation_name or f"{func_to_monitor.__name__}"

            # Start monitoring
            metrics = _global_runtime_monitor.start_operation(
                op_name,
                function_name=func_to_monitor.__name__,
                module_name=module_name,
                line_number=line_number,
            )

            try:
                result = func_to_monitor(*args, **kwargs)
                _global_runtime_monitor.end_operation(success=True)
                return result
            except Exception as e:
                _global_runtime_monitor.end_operation(
                    success=False, error_message=str(e)
                )
                raise

        return wrapper

    # Handle both @auto_monitor and @auto_monitor(operation_name="...")
    if func is None:
        return decorator
    else:
        return decorator(func)


def update_global_monitor_logger(logger: logging.Logger):
    """Update the global runtime monitor with a specific logger."""
    global _global_runtime_monitor
    _global_runtime_monitor.logger = logger


def update_global_monitor_debug_mode(debug_mode: bool):
    """Update the global runtime monitor debug mode."""
    global _global_runtime_monitor
    _global_runtime_monitor.debug_mode = debug_mode


@contextmanager
def runtime_context(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    debug_mode: bool = False,
):
    """
    Context manager for runtime monitoring of code blocks.

    Usage:
        with runtime_context("my_operation"):
            # Code to monitor
            pass
    """
    monitor = RuntimeMonitor(logger, debug_mode)
    metrics = monitor.start_operation(operation_name)

    try:
        yield metrics
        monitor.end_operation(success=True)
    except Exception as e:
        monitor.end_operation(success=False, error_message=str(e))
        raise


@contextmanager
def command_context(
    command_name: str, logger: Optional[logging.Logger] = None, debug_mode: bool = False
):
    """
    Context manager for monitoring complete commands.

    Usage:
        with command_context("loadpdb"):
            # All operations within this block will be tracked
            # Only summary will be logged unless debug_mode=True
            pass
    """
    monitor = RuntimeMonitor(logger, debug_mode)
    command_metrics = monitor.start_command(command_name)

    try:
        yield command_metrics
        monitor.end_command(success=True)
    except Exception as e:
        monitor.end_command(success=False, error_message=str(e))
        raise


def setup_logger(debug_mode=False, log_file: Optional[str] = None):
    """
    Sets up an enhanced logger for the YAGWIP application with runtime monitoring.

    Parameters:
        debug_mode (bool): If True, sets console logging to DEBUG level; otherwise INFO.
        log_file (str, optional): Custom log file path. If None, generates timestamped filename.

    Returns:
        logging.Logger: Configured logger instance for use throughout the CLI.
    """
    # Create or retrieve the logger with a fixed name
    logger = logging.getLogger("yagwip")

    # Set the logger to the most verbose level to ensure all messages are captured
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplicate logs if the function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler Setup
    ch = logging.StreamHandler()  # Handler for stdout/stderr
    ch_level = (
        logging.DEBUG if debug_mode else logging.INFO
    )  # Use DEBUG level in debug mode
    ch.setLevel(ch_level)

    # Define the log message format for console output
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(formatter)

    # Attach the console handler to the logger
    logger.addHandler(ch)

    # Generate a timestamped filename for the log file if not provided
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"yagwip_{timestamp}.log"

    # Create a file handler to log everything, regardless of debug mode
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # Capture all details regardless of debug mode

    # Enhanced formatter for file logging with more details
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(file_formatter)

    # Attach the file handler to the logger
    logger.addHandler(fh)

    # Log initial system information
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")
        logger.info("=== YAGWIP Runtime Monitoring Started ===")
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(
            f"Initial Memory Usage: {memory.percent:.1f}% ({memory.used / (1024 ** 3):.2f} GB used)"
        )
        logger.info(f"Initial Disk Usage: {disk.percent:.1f}%")
        logger.info(f"CPU Cores: {psutil.cpu_count()}")
        logger.info("=" * 50)
    except ImportError:
        logger.info("psutil not available - limited system monitoring")
    except Exception as e:
        logger.warning(f"Could not initialize system monitoring: {e}")

    # Optional: Notify the user where logs are being written
    if not debug_mode:
        logger.info("Output logged to %s", log_file)
    else:
        logger.debug("Debug logging active; also writing to %s", log_file)

    # Update the global runtime monitor with this logger and debug mode
    update_global_monitor_logger(logger)
    update_global_monitor_debug_mode(debug_mode)

    # Return the configured logger object
    return logger


# Convenience function for runtime monitoring
def monitor_runtime(operation_name: Optional[str] = None):
    """Decorator to monitor runtime of any function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__name__}"
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                print(f"[RUNTIME] {op_name}: {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                print(f"[RUNTIME] {op_name} FAILED: {duration:.2f}s - {str(e)}")
                raise

        return wrapper

    return decorator


# Function to get global runtime summary
def get_global_runtime_summary() -> Dict[str, Any]:
    """Get a summary of all global runtime metrics."""
    return _global_runtime_monitor.get_summary()


# Function to clear global runtime history
def clear_global_runtime_history():
    """Clear the global runtime monitoring history."""
    _global_runtime_monitor.metrics_history.clear()
