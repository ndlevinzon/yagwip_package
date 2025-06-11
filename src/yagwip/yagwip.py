from .build import run_pdb2gmx, run_solvate, run_genions
from pathlib import Path
from .sim import run_em, run_nvt, run_npt, run_production, run_tremd
from .utils import setup_logger, complete_loadpdb, complete_loadgro, insert_itp_into_top_files
from importlib.resources import files
import importlib.metadata
import cmd
import os
import argparse
import sys
import random
import shutil
import re


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
        self.ligand_pdb_path = None
        self.basename = None                                # Base filename (without extension)
        self.print_banner()                                 # Prints intro banner to command line
        self.user_itp_paths = []                            # Stores user input paths for do_source

        # Dictionary of custom command overrides set by the user, not implemented yet
        self.custom_cmds = {
            "pdb2gmx": None,
            "solvate": None,
            "genions": None,
        }

    def default(self, line):
        """Throws error when command is not recognized"""
        print(f"[!] Unknown command: {line}")

    def do_debug(self, arg):
        """
        Debug Mode: Simply prints commands to the command line that
        would have otherwise be executed. Prints to console instead of log

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

        print(f"[DEBUG] Debug mode is now {'ON' if self.debug else 'OFF'}")

    def print_banner(self):
        """
        Prints YAGWIP Banner Logo on Start
        Banner: src/yagwip/assets/banner.txt
        """
        try:
            banner_path = files("yagwip.assets").joinpath("banner.txt")
            with open(banner_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print("[!] Could not load banner:", e)

    def do_show(self, arg):
        """Show current custom or default commands."""
        for k in ["pdb2gmx", "solvate", "genions"]:
            cmd = self.custom_cmds.get(k)
            print(f"{k}: {cmd if cmd else '[DEFAULT]'}")

    def do_set(self, arg):
        """
        Edit the default command string for pdb2gmx, solvate, or genions.

        Usage:
            set pdb2gmx
            set solvate
            set genions

        The user is shown the current command and can modify it inline.
        Press ENTER to accept the modified command.
        Type 'quit' to cancel.
        """
        valid_keys = ["pdb2gmx", "solvate", "genions"]
        cmd_key = arg.strip().lower()

        if cmd_key not in valid_keys:
            print(f"[!] Usage: set <{'|'.join(valid_keys)}>")
            return

        # Get the default command string
        if cmd_key == "pdb2gmx":
            base = self.basename if self.basename else "PLACEHOLDER"
            default = f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        elif cmd_key == "solvate":
            base = self.basename if self.basename else "PLACEHOLDER"
            default = f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro -c -d 1.0 -bt cubic && " \
                      f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs spc216.gro -o {base}.solv.gro -p topol.top"
        elif cmd_key == "genions":
            base = self.basename if self.basename else "PLACEHOLDER"
            ions_mdp = "ions.mdp"  # assuming it's copied to current dir already
            default = f"{self.gmx_path} grompp -f {ions_mdp} -c {base}.solv.gro -r {base}.solv.gro -p topol.top -o ions.tpr && " \
                      f"{self.gmx_path} genion -s ions.tpr -o {base}.solv.ions.gro -p topol.top -pname NA -nname CL -conc 0.150 -neutral"

        # Show current value
        current = self.custom_cmds.get(cmd_key) or default
        print(f"[EDIT {cmd_key}] Current command:\n{current}")
        print("Type new command or press ENTER to keep current. Type 'quit' to cancel.")

        # Prompt user
        new_cmd = input("New command: ").strip()
        if new_cmd.lower() == "quit":
            print("[SET] Edit canceled.")
            return
        elif new_cmd == "":
            self.custom_cmds[cmd_key] = current
            print(f"[SET] Keeping existing command.")
        else:
            self.custom_cmds[cmd_key] = new_cmd
            print(f"[SET] Updated command for {cmd_key}.")

    def complete_loadpdb(self, text, line, begidx, endidx):
        """Adds tab completion for .pdb files for use in loadpdb"""
        return complete_loadpdb(text)

    def do_loadpdb(self, arg):
        """
        Loads .pdb path for further building steps. This command should be run first.
        Usage: "loadpdb X.pdb"
        """
        filename = arg.strip()
        if not filename:
            print("Usage: loadPDB <filename.pdb>")
            return

        full_path = os.path.abspath(filename)
        if not os.path.isfile(full_path):
            print(f"[!] '{filename}' not found.")
            return

        self.current_pdb_path = full_path
        self.basename = os.path.splitext(os.path.basename(full_path))[0]

        print(f"PDB file loaded: {full_path}")

        with open(full_path, 'r') as f:
            lines = f.readlines()

        hetatm_lines = [line for line in lines if line.startswith("HETATM")]

        if hetatm_lines:
            ligand_file = 'ligand.pdb'
            protein_file = 'protein.pdb'
            self.ligand_pdb_path = os.path.abspath(ligand_file)
            self.protein_pdb_path = os.path.abspath(protein_file)

            with open(protein_file, 'w') as prot_out, open(ligand_file, 'w') as lig_out:
                for line in lines:
                    if line.startswith("HETATM"):
                        lig_out.write(line)
                    else:
                        prot_out.write(line)
            print(f"Detected ligand. Split into: {protein_file}, {ligand_file}")
        else:
            self.protein_pdb_path = self.current_pdb_path
            print("No HETATM entries found. Using single PDB for protein.")

    def do_pdb2gmx(self, arg):
        """
        Run pdb2gmx. If ligand is present, treat protein and ligand separately.
        Usage: "pdb2gmx"
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return

        # Build protein topology
        protein_pdb = "protein.pdb" if self.ligand_pdb_path else f"{self.current_pdb_path}"
        base_prot = "protein" if self.ligand_pdb_path else f"{self.current_pdb_path}"
        output_gro = f"{base_prot}.gro"
        topol_top = "topol.top"

        run_pdb2gmx(self.gmx_path, protein_pdb, custom_command=self.custom_cmds["genions"], debug=self.debug,
                    logger=self.logger)

        # If ligand is present, insert it into the .gro and topol.top
        if self.ligand_pdb_path:
            print("[PDB2GMX] Ligand detected. Incorporating ligand into system...")

            ligand_pdb = self.ligand_pdb_path
            ligand_gro = "ligand.gro"

            # Extract crystal coordinates from ligand.pdb into a dummy ligand.gro
            self.replace_coordinates_in_gro(output_gro, ligand_pdb, ligand_gro)

            # Combine ligand and protein
            combined_gro = "complex.gro"
            self.combine_systems(output_gro, ligand_gro, combined_gro)

            # Update topol.top with ITP path and ligand name
            ligand_itp_name = Path(self.user_itp_paths[0]).name if self.user_itp_paths else "ligand.itp"
            self.update_topol_to_include_ligand(topol_top, ligand_itp_name, "LIG")

            print(f"[PDB2GMX] System built: {combined_gro} with updated {topol_top}")

    def replace_coordinates_in_gro(self, template_gro, ligand_pdb, output_gro):
        coords = []
        with open(ligand_pdb) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
        with open(template_gro) as fin, open(output_gro, 'w') as fout:
            header = [next(fin) for _ in range(2)]
            fout.writelines(header)
            for i, line in enumerate(fin):
                if len(coords) <= i:
                    break
                x, y, z = coords[i]
                fout.write(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            box = fin.readline()
            fout.write(box)

    def combine_systems(self, protein_gro, ligand_gro, combined_gro):
        with open(protein_gro) as f:
            prot_lines = f.readlines()
        with open(ligand_gro) as f:
            lig_lines = f.readlines()

        header = prot_lines[:2]
        box_line = prot_lines[-1]
        atom_lines = prot_lines[2:-1] + lig_lines[2:-1]
        total_atoms = len(atom_lines)

        with open(combined_gro, 'w') as out:
            out.writelines(header[:1])
            out.write(f"{total_atoms}\n")
            out.writelines(atom_lines)
            out.write(box_line)

    def update_topol_to_include_ligand(self, top_file, ligand_itp_path, ligand_name):
        with open(top_file, 'r') as f:
            lines = f.readlines()

        with open(top_file, 'w') as f:
            inserted_itp = False
            inserted_mol = False
            for line in lines:
                f.write(line)
                if not inserted_itp and line.strip().startswith('#include') and 'forcefield.itp' in line:
                    f.write(f'#include "{ligand_itp_path}"\n')
                    inserted_itp = True
                if '[ molecules ]' in line:
                    inserted_mol = True
                elif inserted_mol and line.strip() == '':
                    f.write(f'{ligand_name}    1\n')
                    inserted_mol = False
    def do_solvate(self, arg):
        """
        Run solvate with optional custom command override. This command should be run after pdb2gmx.
        Usage: "solvate"
        Other Options: use "set solvate" to override defaults
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_solvate(self.gmx_path, self.basename, custom_command=self.custom_cmds["solvate"], debug=self.debug, logger=self.logger)

    def do_genions(self, arg):
        """
        Run genions with optional custom command override. This command should be run after solvate.
        Usage: "genions"
        Other Options: use "set genions" to override defaults
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_genions(self.gmx_path, self.basename, custom_command=self.custom_cmds["genions"], debug=self.debug, logger=self.logger)

    def do_em(self, arg):
        """
        Runs default energy minimization on the command line
        Usage: "em"
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_em(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_nvt(self, arg):
        """
        Runs default NVT equilibration on the command line
        Usage: "em"
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_nvt(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_npt(self, arg):
        """
        Runs default NPT equilibration on the command line
        Usage: "npt"
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_npt(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def do_production(self, arg):
        """
        Runs default production-phase MD on the command line
        Usage: "production"
        """
        if not self.current_pdb_path and not self.debug:
            print("[!] No PDB loaded.")
            return
        run_production(self.gmx_path, self.basename, arg=arg, debug=self.debug, logger=self.logger)

    def complete_tremd(self, text, line, begidx, endidx):
        """Adds tab completion for .solv.ions.gro for use in TREMD replica calculations"""
        args = line.strip().split()

        # Only complete the filename after 'tremd calc'
        if len(args) >= 2 and args[1] == "calc":
            return complete_loadgro(text)
        return []

    def do_tremd(self, arg):
        """
        Generate a TREMD temperature ladder based on a user-specified .gro file.
        This computes replica exchange temperature ranges using the van der Spoel predictor.

        Usage: "tremd calc X.solv.ions.gro"
        """
        run_tremd(self.gmx_path, self.basename, arg=arg, debug=self.debug)

    def do_source(self, arg):
        """
        Add a custom .itp include to be added to all topol.top files.
        This replaces any existing #include lines not in the user-defined list.

        Usage: source /absolute/path/to/custom.itp
        """
        itp_path = arg.strip()

        if not itp_path.endswith('.itp'):
            print("Error: Must provide a path to a .itp file.")
            return

        if not os.path.isfile(itp_path):
            print(f"Error: File '{itp_path}' not found.")
            return

        # Add new path to list (no duplicates)
        if itp_path not in self.user_itp_paths:
            self.user_itp_paths.append(itp_path)
            print(f"Added custom .itp include: {itp_path}")
        else:
            print(f"[!] Path already in include list: {itp_path}")

        # Apply all includes to all topol.top files
        insert_itp_into_top_files(self.user_itp_paths, root_dir=os.getcwd())

        # Display all tracked includes
        print("\nCurrent custom ITP includes:")
        for p in self.user_itp_paths:
            print(f'  #include "{p}"')

    def do_slurm(self, arg):
        """
        Setup SLURM job scripts.

        Usage:
            slurm md cpu
            slurm md gpu
            slurm tremd cpu
            slurm tremd gpu

        Copies template .mdp files and SLURM script to the current directory based on options.
        """
        args = arg.strip().lower().split()
        if len(args) != 2 or args[0] not in ["md", "tremd"] or args[1] not in ["cpu", "gpu"]:
            print("[!] Usage: slurm <md|tremd> <cpu|gpu>")
            return

        sim_type, hardware = args
        template_dir = files("yagwip.templates")

        # Determine which .mdp file to skip
        exclude = "production.mdp" if sim_type == "tremd" else "remd_template.mdp"

        # Copy only relevant .mdp files
        for f in template_dir.iterdir():
            if f.name.endswith(".mdp") and f.name != exclude:
                shutil.copy(f, os.getcwd())
        print(f"[SLURM] Copied .mdp templates for {sim_type} (excluded: {exclude}).")

        # Copy analysis SLURM file for tremd
        if sim_type == "tremd":
            analysis_slurm = template_dir / "run_tremd_analysis.slurm"
            if analysis_slurm.is_file():
                shutil.copy(analysis_slurm, os.getcwd())
                print("[SLURM] Copied run_tremd_analysis.slurm.")
            else:
                print("[!] run_tremd_analysis.slurm not found in template directory.")

        # Determine input SLURM template
        slurm_tpl_name = f"run_gmx_{sim_type}_{hardware}.slurm"
        slurm_tpl_path = template_dir / slurm_tpl_name

        if not slurm_tpl_path.is_file():
            print(f"[!] SLURM template not found: {slurm_tpl_name}")
            return

        if not self.basename:
            print("[!] No structure loaded. Run `loadpdb <file>` and `genion` first.")
            return

        init_gro = f"{self.basename}.solv.ions"

        try:
            with open(slurm_tpl_path, "r") as f:
                slurm_content = f.read()

            # Replace BASE variable in SLURM script with basename
            slurm_content = re.sub(r'__BASE__', self.basename, slurm_content)

            # Replace init variable in SLURM script
            slurm_content = re.sub(r'init="[^"]+"', f'init="{init_gro}"', slurm_content)

            # Write modified SLURM script
            out_slurm = f"{slurm_tpl_name}"
            with open(out_slurm, "w") as f:
                f.write(slurm_content)

            print(f"[SLURM] Customized SLURM script written: {out_slurm}")
        except Exception as e:
            print(f"[!] Failed to configure SLURM script: {e}")

    def do_quit(self, _):
        """
        Quit the CLI.
        Usage: "quit"
        """
        self.print_random_quote()
        print("Copyright (c) 2025 gregorpatof, NDL\nQuitting YAGWIP.")
        return True

    def print_random_quote(self):
        """
        Prints random quote on exit.
        Quotes: scr/yagwip/assets/quotes.txt
        """
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
