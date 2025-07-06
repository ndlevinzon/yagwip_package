import time
import logging
import psutil
import os
import platform
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum


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


class RuntimeMonitor:
    """Monitors runtime performance and system metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.metrics_history: List[RuntimeMetrics] = []
        self.current_operation: Optional[RuntimeMetrics] = None

    def _get_system_info(self) -> SystemInfo:
        """Capture current system information."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            return SystemInfo(
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                platform=f"{platform.system()} {platform.release()}",
                python_version=platform.python_version(),
                process_id=os.getpid()
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
                process_id=os.getpid()
            )

    def start_operation(self, operation_name: str, **kwargs) -> RuntimeMetrics:
        """Start monitoring an operation."""
        self.current_operation = RuntimeMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            system_info_start=self._get_system_info(),
            additional_info=kwargs
        )

        if self.logger:
            self.logger.info(f"Starting operation: {operation_name}")
            if self.current_operation.system_info_start:
                self._log_system_info("START", self.current_operation.system_info_start)

        return self.current_operation

    def end_operation(self, success: bool = True, error_message: Optional[str] = None) -> Optional[RuntimeMetrics]:
        """End monitoring an operation and calculate metrics."""
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
        if self.current_operation.system_info_start and self.current_operation.system_info_end:
            self.current_operation.memory_delta_mb = (
                    self.current_operation.system_info_end.memory_used_mb -
                    self.current_operation.system_info_start.memory_used_mb
            )
            self.current_operation.cpu_delta_percent = (
                    self.current_operation.system_info_end.cpu_percent -
                    self.current_operation.system_info_start.cpu_percent
            )

        # Log runtime information
        if self.logger:
            self._log_runtime_summary(self.current_operation)

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
            f"  Memory Usage: {system_info.memory_percent:.1f}% ({system_info.memory_used_mb:.1f} MB used, {system_info.memory_available_mb:.1f} MB available)")
        self.logger.info(f"  Disk Usage: {system_info.disk_usage_percent:.1f}%")
        self.logger.info(f"  Platform: {system_info.platform}")
        self.logger.info(f"  Python: {system_info.python_version}")
        self.logger.info(f"  Process ID: {system_info.process_id}")

    def _log_runtime_summary(self, metrics: RuntimeMetrics):
        """Log runtime summary for an operation."""
        if not self.logger:
            return

        status = "SUCCESS" if metrics.success else "FAILED"
        duration_str = f"{metrics.duration_seconds:.2f}s" if metrics.duration_seconds else "unknown"

        self.logger.info(f"[RUNTIME] Operation: {metrics.operation_name}")
        self.logger.info(f"  Status: {status}")
        self.logger.info(f"  Duration: {duration_str}")

        if metrics.memory_delta_mb is not None:
            memory_change = "+" if metrics.memory_delta_mb >= 0 else ""
            self.logger.info(f"  Memory Change: {memory_change}{metrics.memory_delta_mb:.1f} MB")

        if metrics.cpu_delta_percent is not None:
            cpu_change = "+" if metrics.cpu_delta_percent >= 0 else ""
            self.logger.info(f"  CPU Change: {cpu_change}{metrics.cpu_delta_percent:.1f}%")

        if metrics.error_message:
            self.logger.error(f"  Error: {metrics.error_message}")

        # Log additional info if present
        if metrics.additional_info:
            for key, value in metrics.additional_info.items():
                self.logger.info(f"  {key}: {value}")

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
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0
        }


class LoggingMixin:
    """Enhanced mixin class to provide consistent logging functionality with runtime monitoring."""

    def __init__(self):
        self.runtime_monitor = RuntimeMonitor(getattr(self, 'logger', None))

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
                if hasattr(args[0], 'runtime_monitor'):
                    monitor = args[0].runtime_monitor
                else:
                    # Fallback to basic timing
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        logger = getattr(args[0], 'logger', None)
                        if logger:
                            logger.info(f"[RUNTIME] {op_name}: {duration:.2f}s")
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        logger = getattr(args[0], 'logger', None)
                        if logger:
                            logger.error(f"[RUNTIME] {op_name} FAILED: {duration:.2f}s - {str(e)}")
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
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)

    # Attach the file handler to the logger
    logger.addHandler(fh)

    # Log initial system information
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        logger.info("=== YAGWIP Runtime Monitoring Started ===")
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(f"Initial Memory Usage: {memory.percent:.1f}% ({memory.used / (1024 ** 3):.2f} GB used)")
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
