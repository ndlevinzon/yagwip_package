from .utils import run_gromacs_command, LoggingMixin
from importlib.resources import files

# Constants for GROMACS command inputs
PIPE_INPUTS = {
    'pdb2gmx': '1\n',
    'genion_prot': '13\n',
    'genion_complex': '15\n'
}


class Builder(LoggingMixin):
    def __init__(self, gmx_path, debug=False, logger=None):
        self.gmx_path = gmx_path
        self.debug = debug
        self.logger = logger

    def _resolve_basename(self, basename):
        if not basename and not self.debug:
            msg = "[!] No PDB loaded. Use `loadPDB <filename.pdb>` first."
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)
            return None
        return basename if basename else "PLACEHOLDER"

    def run_pdb2gmx(self, basename, custom_command=None):
        base = self._resolve_basename(basename)
        if base is None:
            return

        command = custom_command or (
            f"{self.gmx_path} pdb2gmx -f {base}.pdb -o {base}.gro -water spce -ignh"
        )

        if self.debug:
            print(f"[DEBUG] Command: {command}")
            return

        self._log(f"[#] Running pdb2gmx for {base}.pdb...")
        run_gromacs_command(command, pipe_input=PIPE_INPUTS['pdb2gmx'], debug=self.debug, logger=self.logger)

    def run_solvate(self, basename, arg="", custom_command=None):
        base = self._resolve_basename(basename)
        if base is None:
            return

        default_box = " -c -d 1.0 -bt cubic"
        default_water = "spc216.gro"
        parts = arg.strip().split()
        box_options = parts[0] if len(parts) > 0 else default_box
        water_model = parts[1] if len(parts) > 1 else default_water

        default_cmds = [
            f"{self.gmx_path} editconf -f {base}.gro -o {base}.newbox.gro{box_options}",
            f"{self.gmx_path} solvate -cp {base}.newbox.gro -cs {water_model} -o {base}.solv.gro -p topol.top"
        ]

        if self.debug:
            for cmd in default_cmds:
                print(f"[DEBUG] Command: {cmd}")
            return

        if custom_command:
            self._log("[CUSTOM] Using custom solvate command")
            run_gromacs_command(custom_command, debug=self.debug, logger=self.logger)
        else:
            for cmd in default_cmds:
                run_gromacs_command(cmd, debug=self.debug, logger=self.logger)

    def run_genions(self, basename, custom_command=None):
        base = self._resolve_basename(basename)
        if base is None:
            return

        default_ions = files("yagwip.templates").joinpath("ions.mdp")
        input_gro = f"{base}.solv.gro"
        output_gro = f"{base}.solv.ions.gro"
        tpr_out = "ions.tpr"
        ion_options = "-pname NA -nname CL -conc 0.150 -neutral"
        grompp_opts = ""
        ion_pipe_input = PIPE_INPUTS['genion_prot'] if base.endswith('protein') else PIPE_INPUTS['genion_complex']

        default_cmds = [
            f"{self.gmx_path} grompp -f {default_ions} -c {input_gro} -r {input_gro} -p topol.top -o {tpr_out} {grompp_opts} -maxwarn 50",
            f"{self.gmx_path} genion -s {tpr_out} -o {output_gro} -p topol.top {ion_options}"
        ]

        if self.debug:
            for cmd in default_cmds:
                print(f"[DEBUG] Command: {cmd}")
            return

        self._log(f"[#] Running genion for {base}...")

        if custom_command:
            self._log("[CUSTOM] Using custom genion command")
            run_gromacs_command(custom_command, debug=self.debug, logger=self.logger)
        else:
            for cmd in default_cmds:
                run_gromacs_command(cmd, pipe_input=ion_pipe_input, debug=self.debug, logger=self.logger)


class Modeller(LoggingMixin):
    def __init__(self, pdb, logger=None, debug=False, output_file="protein_test.pdb"):
        self.logger = logger
        self.debug = debug
        self.pdb = pdb
        self.output_file = output_file

    def find_missing_residues(self):
        """
        Identifies missing internal residues by checking for gaps in residue numbering.
        Returns a list of gaps as (chain_id, last_res_before_gap, first_res_after_gap).
        Only uses simple string parsing, no Biopython dependency.
        """
        residue_map = {}  # {chain_id: sorted list of residue IDs}
        with open(self.pdb, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain_id = line[21].strip() or "A"
                    try:
                        res_id = int(line[22:26].strip())
                    except ValueError:
                        continue
                    if chain_id not in residue_map:
                        residue_map[chain_id] = set()
                    residue_map[chain_id].add(res_id)

        gaps = []
        for chain_id, residues in residue_map.items():
            sorted_residues = sorted(residues)
            for i in range(len(sorted_residues) - 1):
                current = sorted_residues[i]
                next_expected = current + 1
                if sorted_residues[i + 1] != next_expected:
                    gaps.append((chain_id, current, sorted_residues[i + 1]))

        self._log(f"[!] Found missing residue ranges: {gaps}" if gaps else "[#] No gaps found.")
        return gaps


class Ligand_Pipeline(LoggingMixin):
    def __init__(self, logger=None, debug=False):
        self.logger = logger
        self.debug = debug

    def convert_pdb_to_mol2(self, pdb_file, mol2_file=None):
        """
        Converts a ligand PDB file to a MOL2 file using a custom parser and writer (no OpenBabel).
        Only ATOM section is supported, no bonds.
        """
        import pandas as pd
        import os
        from datetime import date

        if mol2_file is None:
            mol2_file = pdb_file.replace('.pdb', '.mol2')

        # Efficiently parse ATOM/HETATM lines from PDB
        atom_records = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    atom_id = int(line[6:11])
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    element = line[76:78].strip() if len(line) >= 78 else ''
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26])
                    chain_id = line[21].strip() or 'A'
                    atom_records.append({
                        'atom_id': atom_id,
                        'atom_name': atom_name,
                        'x': x,
                        'y': y,
                        'z': z,
                        'atom_type': element or atom_name[0],
                        'subst_id': 1,
                        'subst_name': res_name,
                        'charge': 0.0,
                        'status_bit': '',
                    })
        if not atom_records:
            self._log(f"[Ligand_Pipeline][ERROR] No atoms found in {pdb_file}.")
            return None
        df_atoms = pd.DataFrame(atom_records)

        # Build minimal MOL2 dict
        mol2 = {}
        mol2['MOLECULE'] = pd.DataFrame([{
            'mol_name': os.path.splitext(os.path.basename(pdb_file))[0],
            'num_atoms': len(df_atoms),
            'num_bonds': 0,
            'num_subst': 1,
            'num_feat': 0,
            'num_sets': 0,
            'mol_type': 'SMALL',
            'charge_type': 'NO_CHARGES',
        }])
        mol2['ATOM'] = df_atoms
        # Bonds can be added here if needed in the future

        # Write MOL2 file
        with open(mol2_file, "w", encoding="utf-8") as out_file:
            out_file.write("###\n")
            today = date.today().strftime("%Y-%m-%d")
            out_file.write(f"### Created by Ligand_Pipeline {today}\n")
            out_file.write("###\n\n")
            out_file.write("@<TRIPOS>MOLECULE\n")
            m = mol2['MOLECULE'].iloc[0]
            out_file.write(f"{m['mol_name']}\n")
            out_file.write(f" {m['num_atoms']} {m['num_bonds']} {m['num_subst']} {m['num_feat']} {m['num_sets']}\n")
            out_file.write(f"{m['mol_type']}\n")
            out_file.write(f"{m['charge_type']}\n\n")
            out_file.write("@<TRIPOS>ATOM\n")
            for _, row in mol2['ATOM'].iterrows():
                out_file.write(
                    f"{int(row['atom_id']):>6d} {row['atom_name']:<8s} {row['x']:>10.4f} {row['y']:>10.4f} {row['z']:>10.4f} {row['atom_type']:<9s} {int(row['subst_id']):<2d} {row['subst_name']:<7s} {row['charge']:>10.4f} {row['status_bit']}\n")
            # No bonds for now
        self._log(f"[Ligand_Pipeline] Atoms: {len(df_atoms)}. MOL2 written to {mol2_file}.")
        return mol2_file
