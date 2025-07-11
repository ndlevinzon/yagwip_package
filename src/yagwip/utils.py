# """
# utils.py -- YAGWIP Utility Functions
# """
#
# # === Standard Library Imports ===
# import os
# import re
# import subprocess
#
# # === Third-Party Imports ===
# import numpy as np
#
# # === Local Imports ===
# from .log_utils import auto_monitor
#
#
# # @auto_monitor
# # def run_gromacs_command(command, pipe_input=None, debug=False, logger=None):
# #     """
# #     Executes a shell command for GROMACS, with optional piping and logging.
# #
# #     Parameters:
# #         command (str): The shell command to execute.
# #         pipe_input (str, optional): Optional string input to pipe into the command (e.g., group selection like "13\n").
# #         debug (bool): If True, prints the command but does not execute it.
# #         logger (Logger, optional): Optional logger to capture stdout, stderr, and execution info.
# #     """
# #     # Log or print the command about to be executed
# #     if logger:
# #         logger.info("[RUNNING] %s", command)
# #     else:
# #         print(f"[RUNNING] {command}")
# #
# #     # In debug mode, skip execution and only log/print a message
# #     if debug:
# #         msg = "[DEBUG MODE] Command not executed."
# #         if logger:
# #             logger.debug(msg)
# #         else:
# #             print(msg)
# #         return
# #
# #     try:
# #         # Execute the command with optional piped input
# #         result = subprocess.run(
# #             command,
# #             input=pipe_input,  # Piped input to stdin, if any (e.g., group number)
# #             shell=True,  # Run command through shell
# #             capture_output=True,  # Capture both stdout and stderr
# #             text=True,  # Decode outputs as strings instead of bytes
# #             check=False,  # Raise an error if the command fails
# #         )
# #
# #         # Strip leading/trailing whitespace from stderr and stdout
# #         stderr = result.stderr.strip()
# #         stdout = result.stdout.strip()
# #         error_text = (
# #             f"{stderr}\n{stdout}".lower()
# #         )  # Combined output used for keyword-based error checks
# #
# #         # Check if the command failed based on return code
# #         if result.returncode != 0:
# #             err_msg = f"[!] Command failed with return code {result.returncode}"
# #
# #             # Log or print error details
# #             if logger:
# #                 logger.error(err_msg)
# #                 if stderr:
# #                     logger.error("[STDERR] %s", stderr)
# #                 if stdout:
# #                     logger.info("[STDOUT] %s", stdout)
# #             print(err_msg)
# #             if stderr:
# #                 print("[STDERR]", stderr)
# #             if stdout:
# #                 print("[STDOUT]", stdout)
# #
# #             # Catch atom number mismatch error
# #             if "number of coordinates in coordinate file" in error_text:
# #                 specific_msg = "[!] Check ligand and protonation: .gro and .top files have different atom counts."
# #                 if logger:
# #                     logger.warning(specific_msg)
# #                 else:
# #                     print(specific_msg)
# #
# #             # Catch periodic improper dihedral type error
# #             elif "no default periodic improper dih. types" in error_text:
# #                 match = re.search(
# #                     r"\[file topol\.top, line (\d+)\]", stderr, re.IGNORECASE
# #                 )
# #                 if match:
# #                     line_num = int(match.group(1))
# #                     top_path = "./topol.top"
# #
# #                     try:
# #                         with open(top_path, "r", encoding="utf-8") as f:
# #                             lines = f.readlines()
# #                         if 0 <= line_num - 1 < len(lines):
# #                             if not lines[line_num - 1].strip().startswith(";"):
# #                                 lines[line_num - 1] = f";{lines[line_num - 1]}"
# #                                 with open(top_path, "w", encoding="utf-8") as f:
# #                                     f.writelines(lines)
# #
# #                                 msg = (
# #                                     f"[#] Detected improper dihedral error, likely an artifact from AMBER forcefields."
# #                                     f" Commenting out line {line_num} in topol.top and rerunning..."
# #                                 )
# #                                 if logger:
# #                                     logger.warning(msg)
# #                                 else:
# #                                     print(msg)
# #
# #                                 # Retry the command
# #                                 retry_msg = (
# #                                     "[#] Rerunning command after modifying topol.top..."
# #                                 )
# #                                 if logger:
# #                                     logger.info(retry_msg)
# #                                 else:
# #                                     print(retry_msg)
# #
# #                                 # Important: recursive retry, but prevent infinite loops
# #                                 return run_gromacs_command(
# #                                     command,
# #                                     pipe_input=pipe_input,
# #                                     debug=debug,
# #                                     logger=logger,
# #                                 )
# #
# #                     except Exception as e:
# #                         fail_msg = f"[!] Failed to modify topol.top: {e}"
# #                         if logger:
# #                             logger.error(fail_msg)
# #                         else:
# #                             print(fail_msg)
# #                 else:
# #                     fallback_msg = "[!] Detected dihedral error, but couldn't find line number in topol.top."
# #                     if logger:
# #                         logger.warning(fallback_msg)
# #                     else:
# #                         print(fallback_msg)
# #
# #             return False  # Final return if not resolved
# #
# #         else:
# #             # If successful, optionally print/log stdout
# #             if stdout:
# #                 if logger:
# #                     logger.info("[STDOUT] %s", stdout)
# #                 else:
# #                     print(stdout)
# #             return True
# #
# #     except Exception as e:
# #         # Catch and log any unexpected runtime exceptions (e.g., permission issues)
# #         if logger:
# #             logger.exception("[!] Failed to run command: %s", e)
# #         else:
# #             print(f"[!] Failed to run command: {e}")
# #         return False
# #
#
# def complete_filename(text, suffix, line=None, begidx=None, endidx=None):
#     """
#     Generic TAB Autocomplete for filenames in the current directory matching a suffix.
#
#     Parameters:
#         text (str): The current input text to match.
#         suffix (str): The file suffix or pattern to match (e.g., ".pdb", "solv.ions.gro").
#     """
#     if not text:
#         return [f for f in os.listdir() if f.endswith(suffix)]
#     return [f for f in os.listdir() if f.startswith(text) and f.endswith(suffix)]
