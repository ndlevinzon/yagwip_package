from .gromacs_sim import GromacsSim
from .logger import setup_logger
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
    print(f"[RUNNING] {command}")
    if debug:
        print("[DEBUG MODE] Command not executed.")
        return

    if pipe_input:
        proc = subprocess.run(shlex.split(command), input=pipe_input.encode(), capture_output=True)
    else:
        proc = subprocess.run(shlex.split(command), capture_output=True)

    if proc.returncode != 0:
        print("[ERROR] Command failed:")
        print(proc.stderr.decode())
    else:
        print(proc.stdout.decode())


class GromacsCLI(cmd.Cmd):
    intro = "Welcome to YAGWIP V0.3. Type help or ? to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        super().__init__()
        self.debug = False
        self.gmx_path = gmx_path
        self.logger = setup_logger(debug_mode=self.debug)
        self.current_pdb_path = None
        self.basename = None
        self.print_banner()
        self.sim = None

        self.custom_commands = {
            "pdb2gmx": None,
            "solvate": None,
            "genion": None,
        }

    def init_sim(self):
        if self.current_pdb_path:
            basedir = os.getcwd()
            self.basename = os.path.splitext(os.path.basename(self.current_pdb_path))[0]
            self.sim = GromacsSim(basedir, self.basename, self.gmx_path, debug_mode=self.debug)
        else:
            print("[!] Issue with gromacs_sim.py!")

    def do_debug(self, arg):
        """
        Debug Mode: Simply prints commands to the command line that
        would have otherwise be executed.

        Usage: Toggle with 'debug', 'debug on', or 'debug off'"
        """

        arg = arg.lower().strip()

        if arg == "on":
            self.debug = True
        elif arg == "off":
            self.debug = False
        else:
            self.debug = not self.debug

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

        print(f"Running pdb2gmx for {self.basename}.pdb...")
        run_gromacs_command(command, pipe_input="15\n", debug=self.debug)

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

    def do_genion(self, arg):
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

        input_gro = f"{base}.solv.gro"
        output_gro = f"{base}.solv.ions.gro"
        tpr_out = "ions.tpr"
        ion_options = "-pname NA -nname CL -conc 0.100 -neutral"
        grompp_opts = ""

        grompp_cmd = f"{self.gmx_path} grompp -f ions.mdp -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts}"
        genion_cmd = f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}"

        print(f"Running genion for {base}...")
        run_gromacs_command(grompp_cmd, debug=self.debug)
        run_gromacs_command(genion_cmd, pipe_input="13\n", debug=self.debug)

    def do_em(self, arg):
        """
        Run energy minimization step.
        Usage:
            em [mdpfile] [suffix] [tprname] [mdrun_suffix]

        Example:
            em minim.mdp .solv.ions em ""
        """
        if not self.sim:
            print("[!] No PDB initialized. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)
        mdpfile = parts[0] if len(parts) > 0 else "minim.mdp"
        suffix = parts[1] if len(parts) > 1 else ".solv.ions"
        tprname = parts[2] if len(parts) > 2 else "em"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        print(f"Running EM with: mdpfile={mdpfile}, suffix={suffix}, tprname={tprname}")
        self.sim.em(mdpfile=mdpfile, suffix=suffix, tprname=tprname, mdrun_suffix=mdrun_suffix)

    def do_nvt(self, arg):
        """
        Run NVT equilibration step.
        Usage:
            nvt [mdpfile] [suffix] [tprname] [mdrun_suffix]

        Example:
            nvt nvt.mdp .em nvt ""
        """
        if not self.sim:
            print("[!] No simulation initialized. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)
        mdpfile = parts[0] if len(parts) > 0 else "nvt.mdp"
        suffix = parts[1] if len(parts) > 1 else ".em"
        tprname = parts[2] if len(parts) > 2 else "nvt"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        print(f"Running NVT with: mdpfile={mdpfile}, suffix={suffix}, tprname={tprname}")
        self.sim.nvt(mdpfile=mdpfile, suffix=suffix, tprname=tprname, mdrun_suffix=mdrun_suffix)

    def do_npt(self, arg):
        """
        Run NPT equilibration step.
        Usage:
            npt [mdpfile] [suffix] [tprname] [mdrun_suffix]

        Example:
            npt npt.mdp .nvt npt ""
        """
        if not self.sim:
            print("[!] No simulation initialized. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)
        mdpfile = parts[0] if len(parts) > 0 else "npt.mdp"
        suffix = parts[1] if len(parts) > 1 else ".nvt"
        tprname = parts[2] if len(parts) > 2 else "npt"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        print(f"Running NPT with: mdpfile={mdpfile}, suffix={suffix}, tprname={tprname}")
        self.sim.npt(mdpfile=mdpfile, suffix=suffix, tprname=tprname, mdrun_suffix=mdrun_suffix)

    def do_production(self, arg):
        """
        Run production simulation.
        Usage:
            production [mdpfile] [inputname] [outname] [mdrun_suffix]
        Example:
            production md1ns.mdp npt. md1ns ""
        """
        if not self.sim:
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
            return

        parts = arg.strip().split(maxsplit=3)
        mdpfile = parts[0] if len(parts) > 0 else "md1ns.mdp"
        inputname = parts[1] if len(parts) > 1 else "npt."
        outname = parts[2] if len(parts) > 2 else "md1ns"
        mdrun_suffix = parts[3] if len(parts) > 3 else ""

        self.sim.production(mdpfile, inputname, outname, mdrun_suffix=mdrun_suffix)

    def do_production_finished(self, arg):
        """
        Check whether the production run finished successfully.
        Usage:
            production_finished [mdname]
        """
        if not self.sim:
            print("No simulation initialized.")
            return

        mdname = arg.strip() if arg.strip() else "md1ns"
        result = self.sim.production_finished(mdname)
        print(f"Production run {'finished' if result else 'not finished'}.")

    def do_prepare_run(self, arg):
        """
        Prepare TPR file for production run.
        Usage:
            prepare_run [mdpfile] [inputname] [outname]
        """
        if not self.sim:
            print("[!] No simulation initialized.")
            return

        parts = arg.strip().split()
        mdpfile = parts[0] if len(parts) > 0 else "md1ns.mdp"
        inputname = parts[1] if len(parts) > 1 else "npt."
        outname = parts[2] if len(parts) > 2 else "md1ns"

        self.sim.prepare_run(mdpfile, inputname, outname)

    def do_convert_production(self, arg):
        """
        Convert trajectory: remove PBC jumps and convert to PDB.
        Usage:
            convert_production <mdname> <pbc_code> <pdb_code>
        """
        if not self.sim:
            print("No simulation initialized.")
            return

        parts = arg.strip().split(maxsplit=2)
        if len(parts) < 3:
            print("Usage: convert_production <mdname> <pbc_code> <pdb_code>")
            return

        mdname, pbc_code, pdb_code = parts
        self.sim.convert_production(mdname, pbc_code, pdb_code)

    def do_quit(self, _):
        """
        Quit the CLI
        """
        self.print_random_quote()
        print("\nCopyright (c) 2025 gregorpatof, NDL\nQuitting YAGWIP.")
        return True

    def print_banner(self):
        try:
            banner_path = files("yagwip.assets").joinpath("banner.txt")
            with open(banner_path, "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            print("[!] Could not load banner:", e)

    def print_random_quote(self):
        try:
            quote_path = files("yagwip.assets").joinpath("quotes.txt")
            with open(quote_path, "r", encoding="utf-8") as f:
                quotes = [line.strip() for line in f if line.strip()]
            if quotes:
                print(f"\nYAGWIP Reminds You...\n{random.choice(quotes)}\n")
        except Exception as e:
            print(f"([!] Unable to load quotes: {e})")

    def default(self, line):
        print(f"[!] Unknown command: {line}")

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
