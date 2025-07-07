"""
Batch Processor for YAGWIP

Handles batch processing of multiple PDB files from different directories
with the same YAGWIP command script execution.
"""
# === Standard Library Imports ===
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import time

# === Local Imports ===
from .base import YagwipBase
from .log import auto_monitor


def _setup_job_logger_worker(log_file: str) -> logging.Logger:
    """Setup logger for individual job in worker process (file only, no console to avoid redundancy)."""
    logger = logging.getLogger(f"batch_job_{Path(log_file).stem}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler only (no console handler to avoid redundant output)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def _execute_single_job_worker_standalone(
    job_data: Dict[str, Any], commands: List[str], gmx_path: str, debug: bool
) -> Dict[str, Any]:
    """
    Standalone worker function for executing a single batch job in a separate process.
    This function runs in a separate process to avoid conflicts.
    """
    # Recreate job object from serialized data
    job = BatchJob.from_dict(job_data)
    job.start_time = datetime.now()
    job.status = "running"

    # Create working directory
    os.makedirs(job.working_dir, exist_ok=True)

    # Setup logging for this job (only file handler, no console to avoid redundancy)
    job_log_file = os.path.join("batch_logs", f"{job.basename}.log")
    job_logger = _setup_job_logger_worker(job_log_file)

    result = {
        "basename": job.basename,
        "pdb_path": job.pdb_path,
        "working_dir": job.working_dir,
        "status": "failed",
        "start_time": job.start_time,
        "end_time": None,
        "duration": None,
        "error_message": None,
        "output_files": [],
        "log_file": job_log_file,
        "job_id": job.job_id,
    }

    try:
        # Copy PDB file to working directory
        working_pdb = os.path.join(job.working_dir, f"{job.basename}.pdb")
        shutil.copy2(job.pdb_path, working_pdb)

        # Change to working directory
        original_cwd = os.getcwd()
        os.chdir(job.working_dir)

        # Execute commands
        for i, command in enumerate(commands, 1):
            job_logger.info(f"Executing command {i}/{len(commands)}: {command}")

            try:
                # Create temporary YAGWIP shell for this job
                from .yagwip import YagwipShell

                yagwip_shell = YagwipShell(gmx_path)
                yagwip_shell.logger = job_logger
                yagwip_shell.debug = debug

                # Modify loadpdb command to use the specific PDB filename
                if command.startswith("loadpdb"):
                    if job.ligand_builder:
                        modified_command = (
                            f"loadpdb {job.basename}.pdb --ligand_builder"
                        )
                    else:
                        modified_command = f"loadpdb {job.basename}.pdb"
                    job_logger.info(f"Modified command: {modified_command}")
                    yagwip_shell.onecmd(modified_command)
                else:
                    # Execute command as-is
                    yagwip_shell.onecmd(command)

                job_logger.info(f"Command {i} completed successfully")

            except Exception as e:
                job_logger.error(f"Command {i} failed: {e}")
                raise

        # Collect output files
        result["output_files"] = _collect_output_files_worker(job.working_dir)
        result["status"] = "completed"

    except Exception as e:
        job_logger.error(f"Job failed: {e}")
        result["error_message"] = str(e)
        result["status"] = "failed"

    finally:
        # Restore original directory
        os.chdir(original_cwd)
        job.end_time = datetime.now()
        result["end_time"] = job.end_time
        result["duration"] = (job.end_time - job.start_time).total_seconds()

        job.status = result["status"]
        job.output_files = result["output_files"]
        job.error_message = result["error_message"]

    return result


def _collect_output_files_worker(working_dir: str) -> List[str]:
    """Collect important output files from working directory in worker process."""
    output_files = []
    important_extensions = [".gro", ".top", ".itp", ".tpr", ".xtc", ".trr", ".edr"]

    for file_path in Path(working_dir).rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() in important_extensions:
                output_files.append(str(file_path.relative_to(working_dir)))

    return output_files


@dataclass
class BatchJob:
    """Represents a single batch job for a PDB file."""

    pdb_path: str
    working_dir: str
    basename: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    output_files: Optional[List[str]] = None
    ligand_builder: bool = False
    job_id: int = 0

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "pdb_path": self.pdb_path,
            "working_dir": self.working_dir,
            "basename": self.basename,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "output_files": self.output_files,
            "ligand_builder": self.ligand_builder,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchJob":
        """Create job from dictionary."""
        # Convert datetime strings back to datetime objects
        start_time = None
        if data.get("start_time"):
            start_time = datetime.fromisoformat(data["start_time"])

        end_time = None
        if data.get("end_time"):
            end_time = datetime.fromisoformat(data["end_time"])

        return cls(
            pdb_path=data["pdb_path"],
            working_dir=data["working_dir"],
            basename=data["basename"],
            status=data["status"],
            start_time=start_time,
            end_time=end_time,
            error_message=data.get("error_message"),
            output_files=data.get("output_files", []),
            ligand_builder=data.get("ligand_builder", False),
            job_id=data.get("job_id", 0),
        )


class ParallelBatchProcessor(YagwipBase):
    """
    Parallel batch processor for handling multiple PDB files with the same YAGWIP commands.

    Features:
    - Process multiple PDBs from different directories
    - Execute the same command script for each PDB
    - Parallel processing with configurable number of workers
    - Isolated working directories for each job
    - Comprehensive logging and error handling
    - Progress tracking and reporting
    - Resume capability for failed jobs
    - Ligand Builder Support: Automatic ligand processing with `--ligand_builder` flag
    """

    def __init__(
        self,
        gmx_path: str = "gmx",
        debug: bool = False,
        logger=None,
        ligand_builder: bool = False,
        max_workers: Optional[int] = None,
    ):
        super().__init__(gmx_path=gmx_path, debug=debug, logger=logger)
        self.jobs: List[BatchJob] = []
        self.batch_config: Dict[str, Any] = {}
        self.results_dir = "batch_results"
        self.logs_dir = "batch_logs"
        self.ligand_builder = ligand_builder

        # Parallel processing settings
        self.max_workers = (
            max_workers if max_workers is not None else min(mp.cpu_count(), 8)
        )  # Cap at 8 workers by default
        self.progress_queue = Queue()
        self.results_lock = threading.Lock()

        # Create results and logs directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self._log_info(
            f"Initialized parallel batch processor with {self.max_workers} workers"
        )

    @auto_monitor
    def load_pdb_list(self, pdb_list_file: str) -> List[BatchJob]:
        """
        Load PDB files from a list file.

        Expected format:
        # Comment lines start with #
        /path/to/protein1.pdb
        /path/to/protein2.pdb
        /path/to/another/protein3.pdb

        Args:
            pdb_list_file: Path to file containing PDB paths

        Returns:
            List of BatchJob objects
        """
        jobs = []

        with open(pdb_list_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Validate PDB file exists
                pdb_path = os.path.abspath(line)
                if not os.path.exists(pdb_path):
                    self._log_warning(
                        f"PDB file not found (line {line_num}): {pdb_path}"
                    )
                    continue

                if not pdb_path.lower().endswith(".pdb"):
                    self._log_warning(
                        f"File doesn't have .pdb extension (line {line_num}): {pdb_path}"
                    )
                    continue

                # Create working directory in the same location as the source PDB
                pdb_dir = os.path.dirname(pdb_path)
                basename = Path(pdb_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                working_dir = os.path.join(pdb_dir, f"{basename}_yagwip_{timestamp}")

                job = BatchJob(
                    pdb_path=pdb_path,
                    working_dir=working_dir,
                    basename=basename,
                    ligand_builder=self.ligand_builder,
                    job_id=len(jobs),
                )
                jobs.append(job)

                self._log_info(f"Added job {job.job_id}: {basename} -> {working_dir}")

        self.jobs = jobs
        self._log_info(f"Loaded {len(jobs)} PDB files for parallel batch processing")
        return jobs

    @auto_monitor
    def load_pdb_directory(
        self, directory_path: str, pattern: str = "*.pdb"
    ) -> List[BatchJob]:
        """
        Load all PDB files from a directory.

        Args:
            directory_path: Directory containing PDB files
            pattern: File pattern to match (default: "*.pdb")

        Returns:
            List of BatchJob objects
        """
        jobs = []
        directory = Path(directory_path)

        if not directory.exists():
            self._log_error(f"Directory not found: {directory_path}")
            return jobs

        pdb_files = list(directory.glob(pattern))

        for pdb_file in pdb_files:
            if pdb_file.is_file():
                # Create working directory in the same location as the source PDB
                pdb_dir = str(pdb_file.parent)
                basename = pdb_file.stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                working_dir = os.path.join(pdb_dir, f"{basename}_yagwip_{timestamp}")

                job = BatchJob(
                    pdb_path=str(pdb_file.absolute()),
                    working_dir=working_dir,
                    basename=basename,
                    ligand_builder=self.ligand_builder,
                    job_id=len(jobs),
                )
                jobs.append(job)

                self._log_info(f"Added job {job.job_id}: {basename} -> {working_dir}")

        self.jobs = jobs
        self._log_info(f"Loaded {len(jobs)} PDB files from {directory_path}")
        return jobs

    @auto_monitor
    def load_pdb_paths(self, pdb_paths: List[str]) -> List[BatchJob]:
        """
        Load PDB files from a list of paths.

        Args:
            pdb_paths: List of PDB file paths

        Returns:
            List of BatchJob objects
        """
        jobs = []

        for pdb_path in pdb_paths:
            pdb_path = os.path.abspath(pdb_path)

            if not os.path.exists(pdb_path):
                self._log_warning(f"PDB file not found: {pdb_path}")
                continue

            if not pdb_path.lower().endswith(".pdb"):
                self._log_warning(f"File doesn't have .pdb extension: {pdb_path}")
                continue

            # Create working directory in the same location as the source PDB
            pdb_dir = os.path.dirname(pdb_path)
            basename = Path(pdb_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            working_dir = os.path.join(pdb_dir, f"{basename}_yagwip_{timestamp}")

            job = BatchJob(
                pdb_path=pdb_path,
                working_dir=working_dir,
                basename=basename,
                ligand_builder=self.ligand_builder,
                job_id=len(jobs),
            )
            jobs.append(job)

            self._log_info(f"Added job {job.job_id}: {basename} -> {working_dir}")

        self.jobs = jobs
        self._log_info(f"Loaded {len(jobs)} PDB files for parallel batch processing")
        return jobs

    @auto_monitor
    def execute_batch_parallel(
        self, command_script: str, resume: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the same YAGWIP command script for all PDB files using parallel processing.

        Args:
            command_script: Path to YAGWIP command script file
            resume: Whether to resume from previous batch run

        Returns:
            Dictionary with batch execution results
        """
        if not self.jobs:
            self._log_error("No jobs loaded. Use load_pdb_* methods first.")
            return {}

        if not os.path.exists(command_script):
            self._log_error(f"Command script not found: {command_script}")
            return {}

        # Load command script
        commands = self._load_command_script(command_script)
        if not commands:
            self._log_error("No valid commands found in script")
            return {}

        # Load previous results if resuming
        previous_results = {}
        if resume:
            previous_results = self._load_previous_results()

        # Filter out completed jobs if resuming
        jobs_to_process = []
        for job in self.jobs:
            if resume and job.basename in previous_results:
                if previous_results[job.basename]["status"] == "completed":
                    self._log_info(f"Skipping completed job: {job.basename}")
                    continue
            jobs_to_process.append(job)

        if not jobs_to_process:
            self._log_info("All jobs already completed.")
            return self._get_completed_results(previous_results)

        # Execute jobs in parallel
        results = {
            "start_time": datetime.now(),
            "total_jobs": len(self.jobs),
            "completed_jobs": 0,
            "failed_jobs": 0,
            "job_results": [],
            "parallel_workers": self.max_workers,
        }

        # Start progress monitoring thread
        progress_thread = threading.Thread(
            target=self._monitor_progress, args=(len(jobs_to_process), results)
        )
        progress_thread.daemon = True
        progress_thread.start()

        # Execute jobs in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs with serializable data
            future_to_job = {
                executor.submit(
                    _execute_single_job_worker_standalone,
                    job.to_dict(),
                    commands,
                    self.gmx_path or "gmx",
                    self.debug,
                ): job
                for job in jobs_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    job_result = future.result()
                    with self.results_lock:
                        results["job_results"].append(job_result)
                        if job_result["status"] == "completed":
                            results["completed_jobs"] += 1
                        else:
                            results["failed_jobs"] += 1

                        # Update job status
                        job.status = job_result["status"]
                        job.output_files = job_result["output_files"]
                        job.error_message = job_result["error_message"]
                        job.start_time = job_result["start_time"]
                        job.end_time = job_result["end_time"]

                        self._log_info(
                            f"Job {job.job_id} ({job.basename}) completed: {job_result['status']}"
                        )

                except Exception as e:
                    self._log_error(
                        f"Job {job.job_id} ({job.basename}) failed with exception: {e}"
                    )
                    with self.results_lock:
                        results["failed_jobs"] += 1
                        job.status = "failed"
                        job.error_message = str(e)

        # Wait for progress thread to finish
        progress_thread.join(timeout=5)

        results["end_time"] = datetime.now()
        results["duration"] = (
            results["end_time"] - results["start_time"]
        ).total_seconds()

        # Save results
        self._save_batch_results(results)

        # Print summary
        self._print_batch_summary(results)

        return results

    def _monitor_progress(self, total_jobs: int, results: Dict[str, Any]):
        """Monitor and report progress of parallel batch processing."""
        start_time = time.time()
        last_report = start_time

        while True:
            with self.results_lock:
                completed = results["completed_jobs"]
                failed = results["failed_jobs"]
                total_completed = completed + failed

                if total_completed >= total_jobs:
                    break

                # Report progress every 30 seconds
                current_time = time.time()
                if current_time - last_report >= 30:
                    elapsed = current_time - start_time
                    rate = total_completed / elapsed if elapsed > 0 else 0
                    remaining = (total_jobs - total_completed) / rate if rate > 0 else 0

                    self._log_info(
                        f"Progress: {total_completed}/{total_jobs} jobs completed "
                        f"({completed} successful, {failed} failed). "
                        f"Rate: {rate:.2f} jobs/min. "
                        f"Estimated time remaining: {remaining/60:.1f} minutes"
                    )
                    last_report = current_time

            time.sleep(5)  # Check every 5 seconds

    def _get_completed_results(
        self, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get results for already completed jobs."""
        results = {"start_time": datetime.now(), "total_jobs": len(self.jobs), "completed_jobs": len(previous_results),
                   "failed_jobs": 0, "job_results": list(previous_results.values()),
                   "parallel_workers": self.max_workers, "end_time": datetime.now(), "duration": 0}

        return results

    @auto_monitor
    def execute_batch(
        self, command_script: str, resume: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the same YAGWIP command script for all PDB files.
        This is a wrapper for the parallel execution.

        Args:
            command_script: Path to YAGWIP command script file
            resume: Whether to resume from previous batch run

        Returns:
            Dictionary with batch execution results
        """
        return self.execute_batch_parallel(command_script, resume)

    def _load_command_script(self, script_path: str) -> List[str]:
        """Load and validate YAGWIP commands from script file."""
        commands = []

        with open(script_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Validate command
                if self._is_valid_yagwip_command(line):
                    commands.append(line)
                else:
                    self._log_warning(f"Invalid command (line {line_num}): {line}")

        return commands

    def _is_valid_yagwip_command(self, command: str) -> bool:
        """Check if command is a valid YAGWIP command."""
        valid_commands = {
            "loadpdb",
            "pdb2gmx",
            "solvate",
            "genions",
            "em",
            "nvt",
            "npt",
            "production",
            "tremd",
            "source",
            "slurm",
            "debug",
            "show",
            "runtime",
        }

        cmd_parts = command.split()
        if not cmd_parts:
            return False

        return cmd_parts[0].lower() in valid_commands

    def _setup_job_logger(self, log_file: str) -> logging.Logger:
        """Setup logger for individual job (file only, no console to avoid redundancy)."""
        logger = logging.getLogger(f"batch_job_{Path(log_file).stem}")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler only (no console handler to avoid redundant output)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def _collect_output_files(self, working_dir: str) -> List[str]:
        """Collect important output files from working directory."""
        output_files = []
        important_extensions = [".gro", ".top", ".itp", ".tpr", ".xtc", ".trr", ".edr"]

        for file_path in Path(working_dir).rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in important_extensions:
                    output_files.append(str(file_path.relative_to(working_dir)))

        return output_files

    def _save_batch_results(self, results: Dict[str, Any]):
        """Save batch results to JSON file."""
        results_file = os.path.join(
            self.results_dir,
            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        # Convert datetime objects to strings for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)

        self._log_info(f"Batch results saved to: {results_file}")

    def _load_previous_results(self) -> Dict[str, Any]:
        """Load previous batch results for resume functionality."""
        results_files = list(Path(self.results_dir).glob("batch_results_*.json"))

        if not results_files:
            return {}

        # Load most recent results file
        latest_file = max(results_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            # Convert back to job results dictionary
            job_results = {}
            for job_result in results.get("job_results", []):
                job_results[job_result["basename"]] = job_result

            self._log_info(f"Loaded previous results from: {latest_file}")
            return job_results

        except Exception as e:
            self._log_warning(f"Could not load previous results: {e}")
            return {}

    def _make_json_serializable(self, obj):
        """Convert datetime objects to strings for JSON serialization."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _print_batch_summary(self, results: Dict[str, Any]):
        """Print batch execution summary."""
        self._log_info("=== Parallel Batch Processing Summary ===")
        self._log_info(f"Total Jobs: {results['total_jobs']}")
        self._log_info(f"Completed: {results['completed_jobs']}")
        self._log_info(f"Failed: {results['failed_jobs']}")
        self._log_info(
            f"Success Rate: {results['completed_jobs'] / results['total_jobs'] * 100:.1f}%"
        )
        self._log_info(f"Total Duration: {results['duration']:.2f} seconds")
        self._log_info(f"Parallel Workers: {results.get('parallel_workers', 'N/A')}")

        if results["failed_jobs"] > 0:
            self._log_info("\nFailed Jobs:")
            for job_result in results["job_results"]:
                if job_result["status"] == "failed":
                    self._log_info(
                        f"  - {job_result['basename']}: {job_result['error_message']}"
                    )

    def get_batch_status(self) -> Dict[str, Any]:
        """Get current batch status."""
        if not self.jobs:
            return {"status": "no_jobs_loaded"}

        completed = sum(1 for job in self.jobs if job.status == "completed")
        failed = sum(1 for job in self.jobs if job.status == "failed")
        running = sum(1 for job in self.jobs if job.status == "running")
        pending = sum(1 for job in self.jobs if job.status == "pending")

        return {
            "total_jobs": len(self.jobs),
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "parallel_workers": self.max_workers,
            "jobs": [
                {"basename": job.basename, "status": job.status, "job_id": job.job_id}
                for job in self.jobs
            ],
        }


# Backward compatibility - alias the parallel processor as the main processor
BatchProcessor = ParallelBatchProcessor
