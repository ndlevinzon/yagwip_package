"""
config.py -- YAGWIP Configuration and Dependency Management System

This module provides a centralized configuration system for YAGWIP that:
- Manages all external dependency paths
- Provides dependency detection and validation
- Handles configuration persistence
- Integrates the ToolChecker functionality
"""

# === Standard Library Imports ===
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import platform

# === Local Imports ===
from utils.log_utils import LoggingMixin, auto_monitor, setup_logger


class DependencyStatus(Enum):
    """Status of external dependencies."""
    AVAILABLE = "available"
    NOT_FOUND = "not_found"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class DependencyInfo:
    """Information about an external dependency."""
    name: str
    executable: str
    required: bool = True
    path: Optional[str] = None
    version: Optional[str] = None
    status: DependencyStatus = DependencyStatus.UNKNOWN
    description: str = ""
    website: str = ""
    citation: str = ""

    def __post_init__(self):
        """Initialize status based on path availability."""
        if self.path and os.path.exists(self.path):
            self.status = DependencyStatus.AVAILABLE
        elif self.path:
            self.status = DependencyStatus.NOT_FOUND
        else:
            self.status = DependencyStatus.UNKNOWN


@dataclass
class YagwipConfig:
    """Centralized configuration for YAGWIP."""

    # === External Dependencies ===
    dependencies: Dict[str, DependencyInfo] = field(default_factory=dict)

    # === GROMACS Configuration ===
    gmx_path: str = "gmx"
    gmx_water_model: str = "spce"
    gmx_force_field: str = "amber14sb"
    gmx_max_warnings: int = 50

    # === Simulation Parameters ===
    default_temperature: float = 300.0
    default_pressure: float = 1.0
    default_timestep: float = 0.002
    default_simulation_time: float = 1000.0  # ps

    # === System Configuration ===
    default_box_distance: float = 1.0
    default_ion_concentration: float = 0.150
    default_ion_names: tuple = ("NA", "CL")

    # === File Paths ===
    template_dir: Optional[str] = None
    output_dir: str = "."
    log_dir: str = "logs"

    # === Debug and Logging ===
    debug_mode: bool = False
    verbose_logging: bool = False
    log_level: str = "INFO"

    # === Performance Settings ===
    parallel_jobs: int = 1
    memory_limit: Optional[int] = None  # MB
    timeout_seconds: int = 300

    def __post_init__(self):
        """Initialize default dependencies."""
        if not self.dependencies:
            self._initialize_default_dependencies()

    def _initialize_default_dependencies(self):
        """Initialize the default dependency list."""
        self.dependencies = {
            "gromacs": DependencyInfo(
                name="GROMACS",
                executable="gmx",
                required=True,
                description="Molecular dynamics simulation package",
                website="https://www.gromacs.org/",
                citation="Abraham, M.J., et al. GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers. SoftwareX 1-2, 19-25 (2015)."
            ),
            "orca": DependencyInfo(
                name="ORCA",
                executable="orca",
                required=False,
                description="Quantum chemistry program for calculations",
                website="https://orcaforum.kofo.mpg.de/",
                citation="Neese, F. Software update: the ORCA program system. WIREs Comput. Mol. Sci. 12, e1606 (2022)."
            ),
            "openmpi": DependencyInfo(
                name="OpenMPI",
                executable="mpirun",
                required=False,
                description="Message passing interface for parallel computing",
                website="https://www.open-mpi.org/",
                citation="Gabriel, E., et al. Open MPI: Goals, concept, and design of a next generation MPI implementation. In Recent Advances in Parallel Virtual Machine and Message Passing Interface, 97-104 (2004)."
            ),
            "amber": DependencyInfo(
                name="AmberTools",
                executable="parmchk2",
                required=False,
                description="Molecular mechanics force field and simulation package",
                website="https://ambermd.org/",
                citation="Case, D.A., et al. AMBER 2021. University of California, San Francisco (2021)."
            ),
            "openbabel": DependencyInfo(
                name="OpenBabel",
                executable="obabel",
                required=False,
                description="Chemical toolbox for molecular modeling",
                website="https://openbabel.org/",
                citation="O'Boyle, N.M., et al. Open Babel: An open chemical toolbox. J. Cheminform. 3, 33 (2011)."
            ),
            "acpype": DependencyInfo(
                name="ACPYPE",
                executable="acpype",
                required=False,
                description="AnteChamber PYthon Parser interfacE for topology generation",
                website="https://github.com/alanwilter/acpype",
                citation="Sousa da Silva, A.W. & Vranken, W.F. ACPYPE - AnteChamber PYthon Parser interfacE. BMC Res. Notes 5, 367 (2012)."
            )
        }

    def set_dependency_path(self, name: str, path: str):
        """Set the path for a specific dependency."""
        if name in self.dependencies:
            self.dependencies[name].path = path
            if os.path.exists(path):
                self.dependencies[name].status = DependencyStatus.AVAILABLE
            else:
                self.dependencies[name].status = DependencyStatus.NOT_FOUND

    def get_dependency_path(self, name: str) -> Optional[str]:
        """Get the path for a specific dependency."""
        if name in self.dependencies:
            return self.dependencies[name].path
        return None

    def get_dependency_status(self, name: str) -> DependencyStatus:
        """Get the status of a specific dependency."""
        if name in self.dependencies:
            return self.dependencies[name].status
        return DependencyStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = asdict(self)
        # Convert dependencies to serializable format
        config_dict["dependencies"] = {
            name: {
                "name": dep.name,
                "executable": dep.executable,
                "required": dep.required,
                "path": dep.path,
                "version": dep.version,
                "status": dep.status.value,
                "description": dep.description,
                "website": dep.website,
                "citation": dep.citation
            }
            for name, dep in self.dependencies.items()
        }
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'YagwipConfig':
        """Create configuration from dictionary."""
        # Handle dependencies separately
        dependencies = {}
        if "dependencies" in config_dict:
            for name, dep_dict in config_dict["dependencies"].items():
                dep_info = DependencyInfo(
                    name=dep_dict["name"],
                    executable=dep_dict["executable"],
                    required=dep_dict["required"],
                    path=dep_dict.get("path"),
                    version=dep_dict.get("version"),
                    status=DependencyStatus(dep_dict.get("status", "unknown")),
                    description=dep_dict.get("description", ""),
                    website=dep_dict.get("website", ""),
                    citation=dep_dict.get("citation", "")
                )
                dependencies[name] = dep_info
            del config_dict["dependencies"]

        # Create config instance
        config = cls(**config_dict)
        config.dependencies = dependencies
        return config


class ToolChecker(LoggingMixin):
    """
    Enhanced tool checker integrated with configuration system.

    This class provides comprehensive dependency detection and validation
    for all external tools required by YAGWIP.
    """

    def __init__(self, config: YagwipConfig, logger=None, debug=False):
        """Initialize ToolChecker with configuration."""
        super().__init__()
        self.config = config
        self.logger = logger or setup_logger(debug_mode=debug)
        self.debug = debug

    @auto_monitor
    def check_all_dependencies(self) -> Dict[str, DependencyInfo]:
        """
        Check all external dependencies and update configuration.

        Returns:
            Dictionary of dependency information with updated status
        """
        self._log_info("Checking all external dependencies...")

        for dep_name, dep_info in self.config.dependencies.items():
            self._log_debug(f"Checking {dep_name}...")
            path = self._detect_dependency(dep_info)

            if path:
                self.config.set_dependency_path(dep_name, path)
                version = self._get_dependency_version(dep_name, path)
                if version:
                    dep_info.version = version
                self._log_success(f"Found {dep_name}: {path}")
                if version:
                    self._log_info(f"  Version: {version}")
            else:
                if dep_info.required:
                    self._log_error(f"Required dependency {dep_name} not found")
                else:
                    self._log_warning(f"Optional dependency {dep_name} not found")

        return self.config.dependencies

    @auto_monitor
    def _detect_dependency(self, dep_info: DependencyInfo) -> Optional[str]:
        """
        Detect individual dependency using intensive HPC-aware strategies.

        Args:
            dep_info: Dependency information

        Returns:
            Path to executable if found, None otherwise
        """
        self._log_debug(f"Intensive detection for {dep_info.name} ({dep_info.executable})")

        # Strategy 1: Check current PATH
        path = shutil.which(dep_info.executable)
        if path:
            self._log_debug(f"Found in PATH: {path}")
            return path

        # Strategy 2: Check environment variables (multiple variations)
        env_vars = [
            f"{dep_info.name.upper()}_PATH",
            f"{dep_info.name.upper()}_HOME",
            f"{dep_info.name.upper()}_DIR",
            f"{dep_info.executable.upper()}_PATH",
            f"{dep_info.executable.upper()}_HOME"
        ]

        for env_var in env_vars:
            env_path = os.environ.get(env_var)
            if env_path:
                # Check if it's a directory or file
                if os.path.isdir(env_path):
                    potential_path = os.path.join(env_path, dep_info.executable)
                    if os.path.exists(potential_path) and os.access(potential_path, os.X_OK):
                        self._log_debug(f"Found via env var {env_var}: {potential_path}")
                        return potential_path
                elif os.path.isfile(env_path) and os.access(env_path, os.X_OK):
                    self._log_debug(f"Found via env var {env_var}: {env_path}")
                    return env_path

        # Strategy 3: Check module system (Lmod/Environment Modules)
        module_path = self._check_module_system(dep_info)
        if module_path:
            self._log_debug(f"Found via module system: {module_path}")
            return module_path

        # Strategy 4: Check common installation directories
        common_paths = self._get_common_paths(dep_info.name)
        for common_path in common_paths:
            if os.path.exists(common_path) and os.access(common_path, os.X_OK):
                self._log_debug(f"Found in common path: {common_path}")
                return common_path

        # Strategy 5: Check HPC-specific paths
        hpc_paths = self._get_hpc_paths(dep_info)
        for hpc_path in hpc_paths:
            if os.path.exists(hpc_path) and os.access(hpc_path, os.X_OK):
                self._log_debug(f"Found in HPC path: {hpc_path}")
                return hpc_path

        # Strategy 6: Interactive shell probing
        shell_path = self._probe_shell_environment(dep_info)
        if shell_path:
            self._log_debug(f"Found via shell probe: {shell_path}")
            return shell_path

        # Strategy 7: Extended PATH checking with subprocess
        extended_path = self._check_extended_path(dep_info)
        if extended_path:
            self._log_debug(f"Found via extended PATH: {extended_path}")
            return extended_path

        self._log_debug(f"No path found for {dep_info.name}")
        return None

    def _get_common_paths(self, tool_name: str) -> List[str]:
        """Get common installation paths for a tool."""
        system = platform.system().lower()

        common_paths = {
            "gromacs": [
                "/usr/local/gromacs/bin/gmx",
                "/opt/gromacs/bin/gmx",
                "/usr/bin/gmx",
                "C:/Program Files/GROMACS/bin/gmx.exe",
                "C:/GROMACS/bin/gmx.exe"
            ],
            "orca": [
                "/opt/orca/orca",
                "/usr/local/bin/orca",
                "/usr/bin/orca",
                "C:/Program Files/ORCA/orca.exe",
                "C:/ORCA/orca.exe"
            ],
            "openmpi": [
                "/usr/local/bin/mpirun",
                "/usr/bin/mpirun",
                "/opt/openmpi/bin/mpirun",
                "C:/Program Files/OpenMPI/bin/mpirun.exe"
            ],
            "amber": [
                "/usr/local/amber/bin/parmchk2",
                "/opt/amber/bin/parmchk2",
                "/usr/bin/parmchk2",
                "C:/Program Files/AmberTools/bin/parmchk2.exe"
            ],
            "openbabel": [
                "/usr/local/bin/obabel",
                "/usr/bin/obabel",
                "/opt/openbabel/bin/obabel",
                "C:/Program Files/OpenBabel/bin/obabel.exe"
            ],
            "acpype": [
                "/usr/local/bin/acpype",
                "/usr/bin/acpype",
                "/opt/acpype/bin/acpype",
                "C:/Program Files/ACPYPE/acpype.exe"
            ]
        }

        return common_paths.get(tool_name.lower(), [])

    def _check_module_system(self, dep_info: DependencyInfo) -> Optional[str]:
        """Check if dependency is available through module system (Lmod/Environment Modules)."""
        try:
            # Try to get module path using 'which' after loading module
            module_names = [
                dep_info.name.lower(),
                dep_info.executable,
                f"{dep_info.name.lower()}/{dep_info.executable}",
                f"{dep_info.executable}/{dep_info.name.lower()}"
            ]

            for module_name in module_names:
                try:
                    # Try to get the path using module show
                    result = subprocess.run(
                        ["module", "show", module_name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        # Parse module show output to find PATH additions
                        for line in result.stdout.split('\n'):
                            if 'PATH' in line and '=' in line:
                                path_part = line.split('=')[1].strip()
                                potential_path = os.path.join(path_part, dep_info.executable)
                                if os.path.exists(potential_path) and os.access(potential_path, os.X_OK):
                                    return potential_path
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

        except Exception as e:
            self._log_debug(f"Module system check failed: {e}")

        return None

    def _get_hpc_paths(self, dep_info: DependencyInfo) -> List[str]:
        """Get HPC-specific installation paths."""
        hpc_paths = []

        # Common HPC installation directories
        hpc_dirs = [
            "/opt/apps",
            "/usr/local/apps",
            "/sw",
            "/apps",
            "/opt/software",
            "/usr/local/software",
            "/opt/modules",
            "/usr/local/modules",
            "/opt/packages",
            "/usr/local/packages"
        ]

        for hpc_dir in hpc_dirs:
            if os.path.exists(hpc_dir):
                # Look for tool-specific directories
                tool_dirs = [
                    os.path.join(hpc_dir, dep_info.name.lower()),
                    os.path.join(hpc_dir, dep_info.executable),
                    os.path.join(hpc_dir, dep_info.name.upper()),
                    os.path.join(hpc_dir, dep_info.executable.upper())
                ]

                for tool_dir in tool_dirs:
                    if os.path.exists(tool_dir):
                        # Look for bin directory
                        bin_dir = os.path.join(tool_dir, "bin")
                        if os.path.exists(bin_dir):
                            potential_path = os.path.join(bin_dir, dep_info.executable)
                            hpc_paths.append(potential_path)

                        # Also check the tool_dir itself
                        potential_path = os.path.join(tool_dir, dep_info.executable)
                        hpc_paths.append(potential_path)

        return hpc_paths

    def _probe_shell_environment(self, dep_info: DependencyInfo) -> Optional[str]:
        try:
            # Try to get path using an interactive shell that sources .bashrc
            shell_commands = [
                f"bash -i -c 'source ~/.bashrc && which {dep_info.executable}'",
                f"bash -i -c 'module load {dep_info.name.lower()} && which {dep_info.executable}'"
            ]
            for cmd in shell_commands:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    path = result.stdout.strip()
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        return path
        except Exception as e:
            self._log_debug(f"Shell environment probe failed: {e}")
        return None

    def _check_extended_path(self, dep_info: DependencyInfo) -> Optional[str]:
        """Check extended PATH using subprocess."""
        try:
            # Get extended PATH from shell
            result = subprocess.run(
                "echo $PATH",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                path_dirs = result.stdout.strip().split(':')
                for path_dir in path_dirs:
                    potential_path = os.path.join(path_dir, dep_info.executable)
                    if os.path.exists(potential_path) and os.access(potential_path, os.X_OK):
                        return potential_path

        except Exception as e:
            self._log_debug(f"Extended PATH check failed: {e}")

        return None

    @auto_monitor
    def _get_dependency_version(self, dep_name: str, path: str) -> Optional[str]:
        """Get version information for a dependency."""
        try:
            if dep_name == "gromacs":
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'GROMACS version' in line:
                            return line.split('version')[1].strip()

            elif dep_name == "orca":
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'ORCA' in line and 'version' in line.lower():
                            return line.strip()

            elif dep_name == "openmpi":
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'Open MPI' in line:
                            return line.strip()

            elif dep_name == "amber":
                result = subprocess.run(
                    [path, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'AmberTools' in line:
                            return line.strip()

            elif dep_name == "openbabel":
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'Open Babel' in line:
                            return line.strip()

            elif dep_name == "acpype":
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'ACPYPE' in line:
                            return line.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        return None

    @auto_monitor
    def validate_required_dependencies(self) -> bool:
        """
        Validate that all required dependencies are available.

        Returns:
            True if all required dependencies are available, False otherwise
        """
        missing_required = []

        for dep_name, dep_info in self.config.dependencies.items():
            if dep_info.required and dep_info.status != DependencyStatus.AVAILABLE:
                missing_required.append(dep_name)

        if missing_required:
            self._log_error(f"Missing required dependencies: {', '.join(missing_required)}")
            return False

        self._log_success("All required dependencies are available")
        return True

    @auto_monitor
    def check_gromacs_available(self, gmx_path: str = None) -> bool:
        """
        Check if GROMACS is available and can be executed.

        Args:
            gmx_path: GROMACS executable path (uses config default if None)

        Returns:
            True if GROMACS is available, False otherwise
        """
        path = gmx_path or self.config.gmx_path

        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    @auto_monitor
    def check_orca_available(self) -> Optional[str]:
        """
        Check if ORCA is available in the system PATH.

        Returns:
            Path to ORCA executable if found, None otherwise
        """
        dep_info = self.config.dependencies.get("orca")
        if not dep_info:
            return None

        path = self._detect_dependency(dep_info)
        if path:
            self.config.set_dependency_path("orca", path)
            self._log_success(f"ORCA executable found: {path}")
            return path
        else:
            self._log_error("ORCA executable not found in PATH")
            return None

    @auto_monitor
    def check_openmpi_available(self) -> Optional[str]:
        """
        Check if OpenMPI (mpirun) is available in the system PATH.

        Returns:
            Path to mpirun executable if found, None otherwise
        """
        dep_info = self.config.dependencies.get("openmpi")
        if not dep_info:
            return None

        path = self._detect_dependency(dep_info)
        if path:
            self.config.set_dependency_path("openmpi", path)
            self._log_success(f"OpenMPI executable found: {path}")
            return path
        else:
            self._log_error("OpenMPI executable not found in PATH")
            return None

    @auto_monitor
    def check_amber_available(self) -> Optional[str]:
        """
        Check if AmberTools (parmchk2) is available in the system PATH.

        Returns:
            Path to parmchk2 executable if found, None otherwise
        """
        dep_info = self.config.dependencies.get("amber")
        if not dep_info:
            return None

        path = self._detect_dependency(dep_info)
        if path:
            self.config.set_dependency_path("amber", path)
            self._log_success(f"AmberTools executable found: {path}")
            return path
        else:
            self._log_error("AmberTools executable not found in PATH")
            return None

    @auto_monitor
    def check_openbabel_available(self) -> Optional[str]:
        """
        Check if OpenBabel (obabel) is available in the system PATH.

        Returns:
            Path to obabel executable if found, None otherwise
        """
        dep_info = self.config.dependencies.get("openbabel")
        if not dep_info:
            return None

        path = self._detect_dependency(dep_info)
        if path:
            self.config.set_dependency_path("openbabel", path)
            self._log_success(f"OpenBabel executable found: {path}")
            return path
        else:
            self._log_error("OpenBabel executable not found in PATH")
            return None

    @auto_monitor
    def check_acpype_available(self) -> Optional[str]:
        """
        Check if ACPYPE is available in the system PATH.

        Returns:
            Path to acpype executable if found, None otherwise
        """
        dep_info = self.config.dependencies.get("acpype")
        if not dep_info:
            return None

        path = self._detect_dependency(dep_info)
        if path:
            self.config.set_dependency_path("acpype", path)
            self._log_success(f"ACPYPE executable found: {path}")
            return path
        else:
            self._log_error("ACPYPE executable not found in PATH")
            return None

    def get_dependency_path(self, name: str) -> Optional[str]:
        """Get the path for a specific dependency."""
        return self.config.get_dependency_path(name)

    def print_dependency_status(self):
        """Print a summary of all dependency statuses."""
        self._log_info("=== Dependency Status ===")

        for dep_name, dep_info in self.config.dependencies.items():
            status_icon = "✓" if dep_info.status == DependencyStatus.AVAILABLE else "✗"
            required_marker = " (REQUIRED)" if dep_info.required else " (OPTIONAL)"

            self._log_info(f"{status_icon} {dep_name}{required_marker}")
            if dep_info.path:
                self._log_info(f"    Path: {dep_info.path}")
            if dep_info.version:
                self._log_info(f"    Version: {dep_info.version}")
            if dep_info.description:
                self._log_info(f"    Description: {dep_info.description}")


class ConfigManager:
    """
    Configuration manager for YAGWIP.

    Handles loading, saving, and managing configuration files.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (default: ~/.yagwip/config.json)
        """
        if config_file is None:
            config_dir = Path.home() / ".yagwip"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "config.json"

        self.config_file = Path(config_file)
        self.logger = setup_logger(debug_mode=False)

    def load_config(self) -> YagwipConfig:
        """
        Load configuration from file or create default.

        Returns:
            YagwipConfig instance
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                config = YagwipConfig.from_dict(config_dict)
                self.logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                self.logger.warning(f"Failed to load config from {self.config_file}: {e}")
                self.logger.info("Creating default configuration")

        # Create default configuration
        config = YagwipConfig()
        self.save_config(config)
        return config

    def save_config(self, config: YagwipConfig = None):
        """
        Save configuration to file.

        Args:
            config: Configuration to save (uses loaded config if None)
        """
        if config is None:
            config = self.load_config()

        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def update_config(self, **kwargs):
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration updates
        """
        config = self.load_config()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                self.logger.info(f"Updated {key}: {value}")
            else:
                self.logger.warning(f"Unknown configuration key: {key}")

        self.save_config(config)

    def print_status(self):
        """Print current configuration status."""
        config = self.load_config()

        self.logger.info("=== YAGWIP Configuration Status ===")
        self.logger.info(f"Configuration file: {self.config_file}")
        self.logger.info(f"GROMACS path: {config.gmx_path}")
        self.logger.info(f"Water model: {config.gmx_water_model}")
        self.logger.info(f"Force field: {config.gmx_force_field}")
        self.logger.info(f"Default temperature: {config.default_temperature} K")
        self.logger.info(f"Debug mode: {config.debug_mode}")

        # Print dependency status
        checker = ToolChecker(config, logger=self.logger)
        checker.print_dependency_status()


# === Convenience Functions ===

def get_config() -> YagwipConfig:
    """Get the current configuration."""
    manager = ConfigManager()
    return manager.load_config()


def get_tool_checker() -> ToolChecker:
    """Get a ToolChecker instance with current configuration."""
    config = get_config()
    return ToolChecker(config)


def validate_gromacs_installation(gmx_path: str = None) -> None:
    """
    Validate GROMACS installation and raise an error if not available.

    Args:
        gmx_path: GROMACS executable path

    Raises:
        RuntimeError: If GROMACS is not available or cannot be executed.
    """
    checker = get_tool_checker()
    if not checker.check_gromacs_available(gmx_path):
        raise RuntimeError(
            f"GROMACS ({gmx_path or 'gmx'}) is not available or cannot be executed.\n"
            f"Please ensure GROMACS is installed and available in your PATH.\n"
            f"You can check this by running '{gmx_path or 'gmx'} --version' in your terminal."
        )