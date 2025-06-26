"""
yagtraj.py: Trajectory analysis tool for Gromacs MD simulations

Portions copyright (c) 2025 the Authors.
Authors: Nathan Levinzon, Olivier Mailhot
Contributors:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import cmd
import sys
from .utils import *
from importlib.resources import files
import importlib.metadata

__author__ = "NDL, gregorpatof"
__version__ = importlib.metadata.version("yagwip")


class YAGTRAJ_shell(cmd.Cmd):
    intro = f"Welcome to YAGTRAJ v{__version__}. Type help to list commands."
    prompt = "YAGTRAJ>"

    def __init__(self, gmx_path):
        super().__init__()
        super().__init__()
        self.debug = False  # Toggle debug mode
        self.gmx_path = gmx_path  # Path to GROMACS executable (e.g., "gmx")
        self.logger = setup_logger(debug_mode=self.debug)  # Initialize logging
        self.print_banner()  # Prints intro banner to command line

    def default(self, line):
        """Throws error when command is not recognized"""
        print(f"[!] Unknown command: {line}")

    def print_banner(self):
        """
        Prints YAGTRAJ Banner Logo on Start
        Banner: src/yagwip/assets/yagtraj_banner.txt
        """
        try:
            banner_path = files("yagwip.assets").joinpath("yagtraj_banner.txt")
            with open(str(banner_path), 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print("[!] Could not load banner:", e)

    def print_random_quote(self):
        """
        Prints random quote on exit.
        Quotes: scr/yagwip/assets/quotes.txt
        """
        try:
            quote_path = files("yagwip.assets").joinpath("quotes.txt")
            with open(str(quote_path), "r", encoding="utf-8") as f:
                quotes = [line.strip() for line in f if line.strip()]
            if quotes:
                print(f"\nYAGWIP Reminds You...\n{random.choice(quotes)}\n")
        except Exception as e:
            print(f"([!] Unable to load quotes: {e})")

    def do_quit(self, _):
        """
        Quit the CLI.
        Usage: "quit"
        """
        self.print_random_quote()
        print(f"Copyright (c) 2025 {__author__} \nQuitting YAGWIP.")
        return True


def main():
    parser = argparse.ArgumentParser(description="YAGTRAJ - GROMACS MD Analysis")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run interactive CLI")
    parser.add_argument("-f", "--file", type=str, help="Run commands from input file")

    args = parser.parse_args()
    cli = YAGTRAJ_shell("gmx")

    if args.file:
        # Batch mode: read and execute commands from file
        try:
            with open(args.file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # skip empty lines and comments
                        print(f"YAGTRAJ> {line}")
                        cli.onecmd(line)
        except FileNotFoundError:
            print(f"[!] File '{args.file}' not found.")
            sys.exit(1)
    else:
        # Interactive mode
        cli.cmdloop()


if __name__ == "__main__":
    main()
