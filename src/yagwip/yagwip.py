from src.gromacs_sim import GromacsSim
from logger import setup_logger
import logging
import cmd
import os
import argparse
import sys
import random
import readline

class GromacsCLI(cmd.Cmd):
    intro = "Welcome to YAGWIP V0.2. Type help or ? to list commands."
    prompt = "YAGWIP> "

    def __init__(self, gmx_path):
        super().__init__()
        self.debug = False
        self.gmx_path = gmx_path
        self.logger = setup_logger(debug_mode=self.debug)
        self.current_pdb_path = None
        self.basename = None
        self.sim = None
        self.print_banner("src/assets/banner.txt")  # Show ASCII art at startup

        # Save user modifications to default commands
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
            print("No PDB loaded. Use `loadPDB <filename.pdb>` first.")

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
        """
        Interactively set a custom command to override default GROMACS behavior.
        Usage: set <command_name>
        Example: set solvate
        """
        cmd_name = arg.strip().lower()
        if cmd_name not in ["pdb2gmx", "solvate", "genion"]:
            print("Command not supported for override. Choose from: pdb2gmx, solvate, genion")
            return

        if not self.sim:
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
            return

        # Construct the default command dynamically
        if cmd_name == "pdb2gmx":
            default = self.sim.pdb2gmx("15\n")
        elif cmd_name == "solvate":
            default = self.sim.solvate()
        elif cmd_name == "genion":
            default = self.sim.genion()
        else:
            default = ""

        current = self.custom_commands.get(cmd_name, default)

        # Pre-fill the line using readline
        def prefill():
            readline.insert_text(current)
            readline.redisplay()

        try:
            readline.set_startup_hook(prefill)
            try:
                new_command = input(f"{cmd_name}> ").strip()
            finally:
                readline.set_startup_hook(None)

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

    def do_loadpdb(self, arg):
        """
        Search for a PDB file in the current directory and use it in the
        YAGWIP building pipeline:

        Usage: loadPDB <filename.pdb>
        """
        filename = arg.strip().lower()
        if not filename:
            print("Usage: loadPDB <filename.pdb>")
            return

        full_path = os.path.abspath(filename)

        if os.path.isfile(full_path):
            self.current_pdb_path = full_path
            print(f"PDB file loaded: {full_path}")
            self.init_sim()
        else:
            print(f"Error: '{filename}' not found.")

    def do_pdb2gmx(self, arg):
        """
         Run pdb2gmx with optional "set" modifications
         """
        if not self.sim:
            print("No PDB loaded. Use `loadPDB <filename.pdb>` first.")
            return

        if self.custom_commands["pdb2gmx"]:
            print("[CUSTOM] Using custom pdb2gmx command")
            self.sim.execute([self.custom_commands["pdb2gmx"]])
        else:
            print(f"Running pdb2gmx on {self.basename}.pdb...")
            self.sim.pdb2gmx("15\n")

    def do_solvate(self, arg):
        """
        Run solvate step after pdb2gmx. Usage:
        solvate [box_options] [water_model]

        Example:
        solvate "-c -d 1.2 -bt dodecahedron" tip3p.gro
        """
        if not self.sim:
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
            return

        # If there's a custom solvate command set
        if hasattr(self, "custom_commands") and self.custom_commands.get("solvate"):
            print("[CUSTOM] Using custom solvate command")
            self.sim.execute([self.custom_commands["solvate"]])
            return

        parts = arg.strip().split()
        box_options = parts[0] if len(parts) > 0 else " -c -d 1.0 -bt cubic"
        water_model = parts[1] if len(parts) > 1 else "spc216.gro"

        print(f"Running solvate with box options: '{box_options}' and water model: '{water_model}'")
        self.sim.solvate(box_options=box_options, water_model=water_model)


    def do_genion(self, arg):
        """
        Run genion step to neutralize system.
        Usage:
            genion <index_code> [ion_options] [grompp_options]

        Example:
            genion 13 "\-pname NA -nname CL -conc 0.150 -neutral" ""
        """
        if not self.sim:
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
            return

        # If there's a custom genion command set
        if hasattr(self, "custom_commands") and self.custom_commands.get("genion"):
            print("[CUSTOM] Using custom genion command")
            self.sim.execute([self.custom_commands["genion"]])
            return

        parts = arg.strip().split(maxsplit=2)

        if len(parts) == 0:
            print("Usage: genion <index_code> [ion_options] [grompp_options]")
            return

        sol_code = parts[0]
        ion_options = parts[1] if len(parts) > 1 else " -pname NA -nname CL -conc 0.100 -neutral"
        grompp_options = parts[2] if len(parts) > 2 else ""

        print(
            f"Running genion with index code: '{sol_code}', ion_options: '{ion_options}', grompp_options: '{grompp_options}'")
        self.sim.genion(sol_code, ion_options=ion_options, grompp_options=grompp_options)

    def do_em(self, arg):
        """
        Run energy minimization step.
        Usage:
            em [mdpfile] [suffix] [tprname] [mdrun_suffix]

        Example:
            em minim.mdp .solv.ions em ""
        """
        if not self.sim:
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
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
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
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
            print("No simulation initialized. Use `loadPDB <filename.pdb>` first.")
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
            print("No simulation initialized.")
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
        self.print_random_quote("src/assets/quotes.txt")
        print("Quitting YAGWIP.")
        return True

    def print_random_quote(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                quotes = [line.strip() for line in f if line.strip()]
            if quotes:
                print(f"YAGWIP Reminds You...\n{random.choice(quotes)}")
        except FileNotFoundError:
            print("\n(No quotes file found. Exiting quietly.)")

    def print_banner(self, banner_filepath):
        try:
            with open(banner_filepath, "r", encoding="utf-8") as f:
                print(f.read())
        except FileNotFoundError:
            print("Welcome to GROLEAP (ASCII banner not found)")

    def default(self, line):
        print(f"Unknown command: {line}")

def main():
    parser = argparse.ArgumentParser(description="GROLEAP - GROMACS CLI interface")
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
                        print(f"groLEAP> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    else:
        # Interactive mode
        cli.cmdloop()

if __name__ == "__main__":
    main()
