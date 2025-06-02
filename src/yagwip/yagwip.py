from .build import run_pdb2gmx, run_solvate, run_genions
from .sim import run_em, run_nvt, run_npt, run_production, run_tremd
from .utils import setup_logger, complete_loadpdb
from importlib.resources import files
import importlib.metadata
import cmd
import os
import argparse
import sys
import random
import readline


class GromacsCLI(cmd.Cmd):
    __version__ = importlib.metadata.version("yagwip")

    # Intro message and prompt for the interactive CLI
    intro = f"Welcome to YAGWIP v{__version__}. Type help to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        super().__init__()
        self.debug = False                                  # Toggle debug mode
        self.gmx_path = gmx_path                            # Path to GROMACS executable (e.g., "gmx")
        self.logger = setup_logger(debug_mode=self.debug)   # Initialize logging
        self.current_pdb_path = None                        # Full path to the loaded PDB file
        self.basename = None                                # Base filename (without extension)
        self.print_banner()                                 # Prints intro banner to command line
        self.sim = None                                     # Placeholder for GromacsSim instance

        # Dictionary of custom command overrides set by the user
        self.custom_commands = {
            "pdb2gmx": None,
            "solvate": None,
            "genion": None,
        }

    def default(self, line):
        print(f"[!] Unknown command: {line}")

    def do_debug(self, arg):
        """
        Debug Mode: Simply prints commands to the command line that
        would have otherwise be executed.

        Usage: Toggle with 'debug', 'debug on', or 'debug off'"
        """

        arg = arg.lower().strip()

        # Parse input to determine new debug state
        if arg == "on":
            self.debug = True
        elif arg == "off":
            self.debug = False          # Toggle if no explicit argument
        else:
            self.debug = not self.debug

        # Update logger and simulation mode
        self.logger = setup_logger(debug_mode=self.debug)

        if self.sim:
            if self.debug:
                self.sim.debug_on()
                print("[DEBUG] Debug Mode ON")
            else:
                self.sim.debug_off()
                print("[DEBUG] Debug Mode OFF")
        else:
            print(f"[DEBUG] Debug mode is now {'ON' if self.debug else 'OFF'}")

    def print_banner(self):
        try:
            banner_path = files("yagwip.assets").joinpath("banner.txt")
            with open(banner_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print("[!] Could not load banner:", e)

# TODO: Still need to find a way to implement user customizations elegantly
#     def do_set(self, arg):
#         cmd_name = arg.strip().lower()
#         if cmd_name not in ["pdb2gmx", "solvate", "genion"]:
#             print("[!] Command not supported for override. Choose from: pdb2gmx, solvate, genion")
#             return
#
#         if not self.sim:
#             print("[!] No PDB initialized. Use `loadPDB <filename.pdb>` first.")
#             return
#
#         # Default GROMACS commands from sim
#         default_cmds = {
#             "pdb2gmx": self.sim.get_pdb2gmx_cmd(),
#             "solvate": self.sim.get_solvate_cmd(),
#             "genion": self.sim.get_genion_cmd(),
#         }
#
#         current = self.custom_commands.get(cmd_name, default_cmds[cmd_name])
#
#         # TAB completion (for file path / flags)
#         def completer(text, state):
#             options = [f for f in os.listdir('.') if f.startswith(text)]
#             if state < len(options):
#                 return options[state]
#             return None
#
#         readline.set_completer_delims(' \t\n=')
#         readline.set_completer(completer)
#         readline.parse_and_bind("tab: complete")
#
#         # Pre-fill
#         def prefill():
#             readline.insert_text(current)
#             readline.redisplay()
#
#         readline.set_startup_hook(prefill)
#
#         try:
#             new_command = input(f"{cmd_name}> ")
#             new_command = new_command.strip()
#
#             if new_command.lower() == "quit":
#                 print(f"[{cmd_name.upper()}] Command edit canceled.")
#                 return
#
#             if new_command and new_command != current:
#                 self.custom_commands[cmd_name] = new_command
#                 print(f"[{cmd_name.upper()}] Custom command updated.")
#             else:
#                 print(f"[{cmd_name.upper()}] No changes made.")
#         except KeyboardInterrupt:
#             print("\nCommand entry canceled.")
#         finally:
#             readline.set_startup_hook(None)
#             readline.set_completer(None)

    def tab_complete_loadpdb(self, text, line, begidx, endidx):
        return complete_loadpdb(text)

    def do_loadpdb(self, arg):
        filename = arg.strip()
        if not filename:
            print("Usage: loadPDB <filename.pdb>")
            return

        full_path = os.path.abspath(filename)
        if os.path.isfile(full_path):
            self.current_pdb_path = full_path
            self.basename = os.path.splitext(os.path.basename(full_path))[0]
            self.logger.info(f"Loaded PDB: {full_path}")
            print(f"PDB file loaded: {full_path}")
        else:
            print(f"[!] '{filename}' not found.")

    def do_pdb2gmx(self, arg):
        """
        Run pdb2gmx with optional custom command override. This command should be run after loadpdb.
        Usage: "pdb2gmx"
        Other Options: use "set pdb2gmx" to override defaults
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_pdb2gmx(self.gmx_path, self.basename, self.custom_commands.get("pdb2gmx"), debug=self.debug, logger=self.logger)

    def do_solvate(self, arg):
        """
        Run solvate with optional custom command override. This command should be run after pdb2gmx.
        Usage: "solvate"
        Other Options: use "set solvate" to override defaults
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_solvate(self.gmx_path, self.basename, self.custom_commands.get("solvate"), debug=self.debug, logger=self.logger)

    def do_genions(self, arg):
        """
        Run genions with optional custom command override. This command should be run after solvate.
        Usage: "genions"
        Other Options: use "set genions" to override defaults
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_genions(self.gmx_path, self.basename, self.custom_commands.get("genions"), debug=self.debug, logger=self.logger)

    def do_em(self, arg):
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_em(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_nvt(self, arg):
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_nvt(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_npt(self, arg):
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_npt(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_production(self, arg):
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_production(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_tremd(self, arg):
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_tremd(self.gmx_path, self.basename, self.custom_commands.get("pdb2gmx"), self.debug)

    def do_quit(self, _):
        """
        Quit the CLI
        """
        self.print_random_quote()
        print("Copyright (c) 2025 gregorpatof, NDL\nQuitting YAGWIP.")
        return True

    def print_random_quote(self):
        try:
            quote_path = files("yagwip.assets").joinpath("quotes.txt")
            with open(quote_path, "r", encoding="utf-8") as f:
                quotes = [line.strip() for line in f if line.strip()]
            if quotes:
                print(f"\nYAGWIP Reminds You...\n{random.choice(quotes)}\n")
        except Exception as e:
            print(f"([!] Unable to load quotes: {e})")


def main():
    parser = argparse.ArgumentParser(description="YAGWIP - GROMACS CLI interface")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run interactive CLI")
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")

    args = parser.parse_args()
    cli = GromacsCLI("gmx")

    if args.file:
        # Batch mode: read and execute commands from file
        try:
            with open(args.file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # skip empty lines and comments
                        print(f"YAGWIP> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    else:
        # Interactive mode
        cli.cmdloop()


if __name__ == "__main__":
    main()
