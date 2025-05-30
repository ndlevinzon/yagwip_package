from .parser import count_residues_in_gro
from .logger import setup_logger
from .tremd_calc import tremd_temperature_ladder
from importlib.resources import files
import logging
import cmd
import os
import argparse
import sys
import random
import readline
import subprocess
import shlex


def run_gromacs_command(command, pipe_input=None, debug=False):
    # Print the command to be executed, regardless of mode
    print(f"[RUNNING] {command}")

    # In debug mode, do not execute the command—just return after printing
    if debug:
        print("[DEBUG MODE] Command not executed.")
        return

    # Execute the shell command with optional piped input
    try:
        result = subprocess.run(
            command,
            input=pipe_input,       # Piped input to the command (e.g., "13\n" for genion). Must be a str if text=True
            shell=True,             # Run command through the shell; required for command string parsing
            capture_output=True,    # Capture stdout and stderr for logging
            text=True               # Treat input/output as text (str) instead of bytes
        )

        # Check the return code to determine if the command failed
        if result.returncode != 0:
            print(f"[ERROR] Command failed with return code {result.returncode}")
            print("[STDERR]", result.stderr.strip())    # Print error output from the command
            print("[STDOUT]", result.stdout.strip())    # Sometimes commands print info to stdout even on failure
        else:
            print(result.stdout.strip())                # Print standard output on success

    # Catch and log any Python-side execution errors (e.g., bad shell syntax, missing command)
    except Exception as e:
        print(f"[EXCEPTION] Failed to run command: {e}")


class GromacsCLI(cmd.Cmd):
    # Intro message and prompt for the interactive CLI
    intro = "Welcome to YAGWIP V0.4.5. Type help or ? to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        super().__init__()
        self.debug = False                                  # Toggle debug mode
        self.gmx_path = gmx_path                            # Path to GROMACS executable (e.g., "gmx")
        self.logger = setup_logger(debug_mode=self.debug)   # Full path to the loaded PDB file
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
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

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

    def do_set(self, arg):
        cmd_name = arg.strip().lower()
        if cmd_name not in ["pdb2gmx", "solvate", "genion"]:
            print("[!] Command not supported for override. Choose from: pdb2gmx, solvate, genion")
            return

        if not self.sim:
            print("[!] No PDB initialized. Use `loadPDB <filename.pdb>` first.")
            return

        # Default GROMACS commands from sim
        default_cmds = {
            "pdb2gmx": self.sim.get_pdb2gmx_cmd(),
            "solvate": self.sim.get_solvate_cmd(),
            "genion": self.sim.get_genion_cmd(),
        }

        current = self.custom_commands.get(cmd_name, default_cmds[cmd_name])

        # TAB completion (for file path / flags)
        def completer(text, state):
            options = [f for f in os.listdir('.') if f.startswith(text)]
            if state < len(options):
                return options[state]
            return None

        readline.set_completer_delims(' \t\n=')
        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")

        # Pre-fill
        def prefill():
            readline.insert_text(current)
            readline.redisplay()

        readline.set_startup_hook(prefill)

        try:
            # This should now show:  pdb2gmx> [editable text here]
            new_command = input(f"{cmd_name}> ")
            new_command = new_command.strip()

            if new_command.lower() == "quit":
                print(f"[{cmd_name.upper()}] Command edit canceled.")
                return

            if new_command and new_command != current:
                self.custom_commands[cmd_name] = new_command
                print(f"[{cmd_name.upper()}] Custom command updated.")
            else:
                print(f"[{cmd_name.upper()}] No changes made.")
        except KeyboardInterrupt:
            print("\nCommand entry canceled.")
        finally:
            readline.set_startup_hook(None)
            readline.set_completer(None)

    def do_loadpdb(self, arg):
        filename = arg.strip()
        if not filename:
            print("Usage: loadPDB <filename.pdb>")
            return

        full_path = os.path.abspath(filename)
        if os.path.isfile(full_path):
            self.current_pdb_path = full_path
            self.basename = os.path.splitext(os.path.basename(full_path))[0]
            print(f"PDB file loaded: {full_path}")
        else:
            print(f"Error: '{filename}' not found.")

    def complete_loadpdb(self, text, line, begidx, endidx):
        """Autocomplete PDB filenames in current directory"""
        if not text:
            completions = [f for f in os.listdir() if f.endswith(".pdb")]
        else:
            completions = [f for f in os.listdir() if f.startswith(text) and f.endswith(".pdb")]
        return completions

    def do_pdb2gmx(self, arg):
        """
        Run pdb2gmx with optional custom command set via 'set'.
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        base = self.basename if self.basename else "PLACEHOLDER"

        default_cmd = f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        command = self.custom_commands.get("pdb2gmx") or default_cmd

        print(f"Running pdb2gmx for {self.basename or 'PLACEHOLDER'}.pdb...")
        run_gromacs_command(command, pipe_input="7\n", debug=self.debug)

    def do_solvate(self, arg):
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        base = self.basename if self.basename else "PLACEHOLDER"

        default_box = " -c -d 1.0 -bt cubic"
        default_water = "spc216.gro"
        parts = arg.strip().split()
        box_options = parts[0] if len(parts) > 0 else default_box
        water_model = parts[1] if len(parts) > 1 else default_water

        default_cmds = [
            f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro{box_options}",
            f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs {water_model} -o {base}.solv.gro -p topol.top"
        ]

        if self.custom_commands.get("solvate"):
            print("[CUSTOM] Using custom solvate command")
            run_gromacs_command(self.custom_commands["solvate"], debug=self.debug)
        else:
            for cmd in default_cmds:
                run_gromacs_command(cmd, debug=self.debug)

    def do_genions(self, arg):
        """
        Run genion step to neutralize system.
        Usage:
            genion
            (runs with default index = 13 and standard options unless overridden with `set`)
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        base = self.basename if self.basename else "PLACEHOLDER"

        if self.custom_commands.get("genion"):
            print("[CUSTOM] Using custom genion command")
            run_gromacs_command(self.custom_commands["genion"], pipe_input="13\n", debug=self.debug)
            return

        default_ions = files("yagwip.templates").joinpath("ions.mdp")
        input_gro = f"{base}.solv.gro"
        output_gro = f"{base}.solv.ions.gro"
        tpr_out = "ions.tpr"
        ion_options = "-pname NA -nname CL -conc 0.100 -neutral"
        grompp_opts = ""

        grompp_cmd = f"{self.gmx_path} grompp -f {default_ions} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts}"
        genion_cmd = f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}"

        print(f"Running genion for {base}...")
        run_gromacs_command(grompp_cmd, debug=self.debug)
        run_gromacs_command(genion_cmd, pipe_input="13\n", debug=self.debug)

    def do_em(self, arg):
        """
        Run mdrun energy minimization with optional custom command set via 'set'.
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)

        # Default values
        default_mdp = files("yagwip.templates").joinpath("em.mdp")
        mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
        suffix = parts[1] if len(parts) > 1 else ".solv.ions"
        tprname = parts[2] if len(parts) > 2 else "em"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        base = self.basename if self.basename else "PLACEHOLDER"
        input_gro = f"{base}{suffix}.gro"
        output_gro = f"{tprname}.gro"
        topol = "topol.top"
        tpr_file = f"{tprname}.tpr"

        # Construct GROMACS commands
        grompp_cmd = (
            f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} "
            f"-p {topol} -o {tpr_file}"
        )
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

        print(f"Running energy minimization for {base}...")
        run_gromacs_command(grompp_cmd, debug=self.debug)
        run_gromacs_command(mdrun_cmd, debug=self.debug)

    def do_nvt(self, arg):
        """
        Run NVT equilibration step using GROMACS.
        Usage:
            nvt [mdpfile] [suffix] [tprname] [mdrun_suffix]

        Example:
            nvt nvt.mdp .em nvt ""
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)

        # Defaults
        default_mdp = files("yagwip.templates").joinpath("nvt.mdp")
        mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
        suffix = parts[1] if len(parts) > 1 else ".em"
        tprname = parts[2] if len(parts) > 2 else "nvt"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        base = self.basename if self.basename else "PLACEHOLDER"
        input_gro = f"{base}{suffix}.gro"
        output_gro = f"{tprname}.gro"
        topol = "topol.top"
        tpr_file = f"{tprname}.tpr"

        # Construct GROMACS commands
        grompp_cmd = (
            f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} "
            f"-p {topol} -o {tpr_file}"
        )
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

        print(f"Running NVT equilibration for {base}...")
        run_gromacs_command(grompp_cmd, debug=self.debug)
        run_gromacs_command(mdrun_cmd, debug=self.debug)

    def do_npt(self, arg):
        """
        Run NPT equilibration step using GROMACS.
        Usage:
            npt [mdpfile] [suffix] [tprname] [mdrun_suffix]

        Example:
            npt npt.mdp .nvt npt ""
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)

        # Defaults
        default_mdp = files("yagwip.templates").joinpath("npt.mdp")
        mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
        suffix = parts[1] if len(parts) > 1 else ".nvt"
        tprname = parts[2] if len(parts) > 2 else "npt"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        base = self.basename if self.basename else "PLACEHOLDER"
        input_gro = f"{base}{suffix}.gro"
        output_gro = f"{tprname}.gro"
        topol = "topol.top"
        tpr_file = f"{tprname}.tpr"

        # Construct GROMACS commands
        grompp_cmd = (
            f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} "
            f"-p {topol} -o {tpr_file}"
        )
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {tprname} {mdrun_suffix}"

        print(f"Running NPT equilibration for {base}...")
        run_gromacs_command(grompp_cmd, debug=self.debug)
        run_gromacs_command(mdrun_cmd, debug=self.debug)

    def do_production(self, arg):
        """
        Run production simulation using GROMACS.
        Usage:
            production [mdpfile] [inputname] [outname] [mdrun_suffix]

        Example:
            production md1ns.mdp npt. md1ns ""
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)

        # Default values
        default_mdp = files("yagwip.templates").joinpath("production.mdp")
        mdpfile = parts[0] if len(parts) > 0 else str(default_mdp)
        inputname = parts[1] if len(parts) > 1 else "npt."
        outname = parts[2] if len(parts) > 2 else "md1ns"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        base = self.basename if self.basename else "PLACEHOLDER"
        input_gro = f"{inputname}gro"
        topol = "topol.top"
        tpr_file = f"{outname}.tpr"

        # Construct GROMACS commands
        grompp_cmd = (
            f"{self.gmx_path} grompp -f {mdpfile} -c {input_gro} -r {input_gro} "
            f"-p {topol} -o {tpr_file}"
        )
        mdrun_cmd = f"{self.gmx_path} mdrun -v -deffnm {outname} {mdrun_suffix}"

        print(f"Running production MD for {base}...")
        run_gromacs_command(grompp_cmd, debug=self.debug)
        run_gromacs_command(mdrun_cmd, debug=self.debug)

    def do_tremd(self, arg):
        """
        Usage:
            TREMD calc <filename.gro>

        This command calculates a temperature ladder for Temperature Replica Exchange MD.
        It parses the .gro file to determine number of protein and water residues, then prompts
        the user for Initial Temperature, Final Temperature, and Exchange Probability.
        Output is written to 'temps.txt' as a comma-separated list.
        """
        args = arg.strip().split()
        if len(args) != 2 or args[0].lower() != "calc":
            print("Usage: TREMD calc <filename.gro>")
            return

        gro_path = os.path.abspath(args[1])
        gro_basename = os.path.splitext(os.path.basename(gro_path))[0]
        gro_dir = os.path.dirname(gro_path)

        if not os.path.isfile(gro_path):
            print(f"[ERROR] File not found: {gro_path}")
            return

        try:
            protein_residues, water_residues = count_residues_in_gro(gro_path)
            print(f"[INFO] Found {protein_residues} protein residues and {water_residues} water residues.")
        except Exception as e:
            print(f"[ERROR] Failed to parse .gro file: {e}")
            return

        try:
            Tlow = float(input("Initial Temperature (K): "))
            Thigh = float(input("Final Temperature (K): "))
            Pdes = float(input("Exchange Probability (0 < P < 1): "))
        except ValueError:
            print("[ERROR] Invalid numeric input.")
            return

        if not (0 < Pdes < 1):
            print("[ERROR] Exchange probability must be between 0 and 1.")
            return
        if Thigh <= Tlow:
            print("[ERROR] Final temperature must be greater than initial temperature.")
            return

        try:
            temperatures = tremd_temperature_ladder(
                Tlow=Tlow,
                Thigh=Thigh,
                Pdes=Pdes,
                Nw=water_residues,
                Np=protein_residues,
                Hff=0,
                Vs=0,
                PC=1,
                WC=0,
                Tol=0.0005
            )
            output = ", ".join(f"{t:.2f}" for t in temperatures)
            output_file = os.path.join(gro_dir, f"{gro_basename}_temps.txt")

            with open(output_file, "w") as f:
                f.write(output + "\n")

            print(f"[SUCCESS] Temperature ladder written to {output_file}:\n{output}")
        except Exception as e:
            print(f"[ERROR] Temperature calculation failed: {e}")

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
