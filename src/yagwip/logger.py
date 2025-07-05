import time
import logging


class LoggingMixin:
    """Mixin class to provide consistent logging functionality across all classes."""

    def _log(self, msg):
        """Log message using logger or print if no logger available."""
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(msg)
        else:
            print(msg)


def setup_logger(debug_mode=False):
    """
    Sets up a logger for the YAGWIP application.

    Parameters:
        debug_mode (bool): If True, sets console logging to DEBUG level; otherwise INFO.

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

    # Generate a timestamped filename for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logfile = f"yagwip_{timestamp}.log"

    # Create a file handler to log everything, regardless of debug mode
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # Capture all details regardless of debug mode
    fh.setFormatter(formatter)

    # Attach the file handler to the logger
    logger.addHandler(fh)

    # Optional: Notify the user where logs are being written
    if not debug_mode:
        logger.info("Output logged to %s", logfile)
    else:
        logger.debug("Debug logging active; also writing to %s", logfile)

    # Return the configured logger object
    return logger
