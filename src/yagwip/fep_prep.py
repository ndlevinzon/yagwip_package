import os
import argparse
import numpy as np
from shutil import copyfile
from collections import defaultdict
from importlib.resources import files


# --- MOL2 parsing ---
def parse_mol2_coords(filename):
    coords = {}
    names = {}
    with open(filename) as f:
        lines = f.readlines()
    in_atoms = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if in_atoms:
            if line.startswith("@<TRIPOS>"):
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            name = parts[1]
            x, y, z = map(float, parts[2:5])
            coords[idx] = (x, y, z)
            names[idx] = name
    return coords, names


# --- PDB parsing ---
def parse_pdb_coords(filename):
    coords = {}
    names = {}
    atom_lines = []
    idx = 1
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords[idx] = (x, y, z)
            names[idx] = line[12:16].strip()
            atom_lines.append(line)
            idx += 1
    return coords, names, atom_lines, lines


def write_aligned_pdb(atom_lines, coords, out_file):
    with open(out_file, 'w') as f:
        for line, (x, y, z) in zip(atom_lines, coords):
            newline = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            f.write(newline)


# --- GRO parsing ---
def parse_gro_coords(filename):
    coords = {}
    names = {}
    atom_lines = []
    with open(filename) as f:
        lines = f.readlines()
    # Skip title and atom count lines
    for i, line in enumerate(lines[2:], 2):
        if len(line.strip()) == 0:
            break
        if len(line) >= 44:  # GRO format: atom index, residue name, atom name, coordinates
            try:
                atom_idx = int(line[15:20])
                atom_name = line[10:15].strip()
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                coords[atom_idx] = (x, y, z)
                names[atom_idx] = atom_name
                atom_lines.append(line)
            except (ValueError, IndexError):
                continue
    return coords, names, atom_lines, lines


def write_aligned_gro(atom_lines, coords, out_file):
    with open(out_file, 'w') as f:
        for line, (x, y, z) in zip(atom_lines, coords):
            newline = line[:20] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[44:]
            f.write(newline)


# --- Atom and Bond classes for MolGraph ---
class Atom:
    def __init__(self, idx, element):
        self.idx = idx
        self.element = element
        self.neighbors = set()
        self.degree = 0


class Bond:
    def __init__(self, a1, a2, order):
        self.a1 = a1
        self.a2 = a2
        self.order = order


# --- MolGraph and MCS utilities ---
class MolGraph:
    def __init__(self):
        self.atoms = {}  # idx: Atom
        self.bonds = []  # list of Bond
        self.adj = defaultdict(set)  # idx: set of neighbor idxs
        self.bond_types = {}  # (min(a1,a2), max(a1,a2)): order

    @staticmethod
    def from_mol2(filename):
        atoms = {}
        bonds = []
        adj = defaultdict(set)
        bond_types = {}
        with open(filename) as f:
            lines = f.readlines()
        atom_section = False
        bond_section = False
        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                atom_section = True
                bond_section = False
                continue
            if line.startswith("@<TRIPOS>BOND"):
                atom_section = False
                bond_section = True
                continue
            if line.startswith("@<TRIPOS>"):
                atom_section = False
                bond_section = False
            if atom_section:
                parts = line.split()
                if len(parts) < 6:
                    continue
                idx = int(parts[0])
                element = ''.join(filter(str.isalpha, parts[5])).upper()
                atoms[idx] = Atom(idx, element)
            if bond_section:
                parts = line.split()
                if len(parts) < 4:
                    continue
                a1 = int(parts[1])
                a2 = int(parts[2])
                order = parts[3]
                bonds.append(Bond(a1, a2, order))
                adj[a1].add(a2)
                adj[a2].add(a1)
                bond_types[(min(a1, a2), max(a1, a2))] = order
        for idx, atom in atoms.items():
            atom.neighbors = adj[idx]
            atom.degree = len(adj[idx])
        g = MolGraph()
        g.atoms = atoms
        g.bonds = bonds
        g.adj = adj
        g.bond_types = bond_types
        return g

    def subgraph(self, atom_indices):
        sg = MolGraph()
        for idx in atom_indices:
            atom = self.atoms[idx]
            sg.atoms[idx] = Atom(idx, atom.element)
        for idx in atom_indices:
            for nbr in self.adj[idx]:
                if nbr in atom_indices:
                    a, b = min(idx, nbr), max(idx, nbr)
                    if (a, b) in sg.bond_types:
                        continue
                    order = self.bond_types[(a, b)]
                    sg.bonds.append(Bond(a, b, order))
                    sg.adj[a].add(b)
                    sg.adj[b].add(a)
                    sg.bond_types[(a, b)] = order
        for idx in sg.atoms:
            sg.atoms[idx].neighbors = sg.adj[idx]
            sg.atoms[idx].degree = len(sg.adj[idx])
        return sg


def enumerate_connected_subgraphs(graph, size):
    results = set()
    for start in graph.atoms:
        stack = [(frozenset([start]), start)]
        while stack:
            nodes, last = stack.pop()
            if len(nodes) == size:
                results.add(nodes)
                continue
            for nbr in graph.atoms[last].neighbors:
                if nbr not in nodes:
                    new_nodes = nodes | {nbr}
                    if len(new_nodes) <= size:
                        stack.append((new_nodes, nbr))
    return [set(s) for s in results]


def find_mcs(g1, g2):
    # Always use the smaller graph for subgraph enumeration
    if len(g1.atoms) > len(g2.atoms):
        g1, g2 = g2, g1
    for size in range(len(g1.atoms), 2, -1):  # Start from largest, need at least 3 for alignment
        subgraphs1 = enumerate_connected_subgraphs(g1, size)
        if not subgraphs1:
            continue
        for atom_indices1 in subgraphs1:
            sg1 = g1.subgraph(atom_indices1)
            # Chemical invariant: element counts
            elem_count1 = defaultdict(int)
            for a in sg1.atoms.values():
                elem_count1[a.element] += 1
            # Find candidate subgraphs in g2 with same element counts
            subgraphs2 = [
                s for s in enumerate_connected_subgraphs(g2, size)
                if all(
                    sum(g2.atoms[i].element == e for i in s) == c
                    for e, c in elem_count1.items()
                )
            ]
            for atom_indices2 in subgraphs2:
                sg2 = g2.subgraph(atom_indices2)
                iso, mapping = are_isomorphic(sg1, sg2)
                if iso:
                    return size, mapping, atom_indices1, atom_indices2
    return 0, None, None, None


def are_isomorphic(g1, g2):
    if len(g1.atoms) != len(g2.atoms):
        return False, None

    def atom_env_hash(graph, idx):
        atom = graph.atoms[idx]
        neighbors = sorted((graph.atoms[n].element, graph.atoms[n].degree) for n in atom.neighbors)
        # Path invariants: elements at distance 2
        dist2 = set()
        for n in atom.neighbors:
            dist2.update(graph.atoms[nn].element for nn in graph.atoms[n].neighbors if nn != idx)
        dist2 = tuple(sorted(dist2))
        return (atom.element, atom.degree, tuple(neighbors), dist2)

    candidates = {}
    for idx1 in g1.atoms:
        hash1 = atom_env_hash(g1, idx1)
        candidates[idx1] = [
            idx2 for idx2 in g2.atoms
            if atom_env_hash(g2, idx2) == hash1
        ]
        if not candidates[idx1]:
            return False, None

    def neighbor_signature(graph, idx):
        return tuple(sorted((graph.atoms[n].element, graph.atoms[n].degree) for n in graph.atoms[idx].neighbors))

    def backtrack(mapping, used2):
        if len(mapping) == len(g1.atoms):
            return True, dict(mapping)
        unmapped = [i for i in g1.atoms if i not in mapping]
        idx1 = min(unmapped, key=lambda i: len([c for c in candidates[i] if c not in used2]))
        for idx2 in candidates[idx1]:
            if idx2 in used2:
                continue
            sig1 = neighbor_signature(g1, idx1)
            sig2 = neighbor_signature(g2, idx2)
            if sig1 != sig2:
                continue
            ok = True
            for nbr1 in g1.atoms[idx1].neighbors:
                if nbr1 in mapping:
                    nbr2 = mapping[nbr1]
                    a, b = min(idx1, nbr1), max(idx1, nbr1)
                    a2, b2 = min(idx2, nbr2), max(idx2, nbr2)
                    if (a2, b2) not in g2.bond_types:
                        ok = False
                        break
                    if g1.bond_types[(a, b)] != g2.bond_types[(a2, b2)]:
                        ok = False
                        break
            if not ok:
                continue
            mapping[idx1] = idx2
            used2.add(idx2)
            found, final_map = backtrack(mapping, used2)
            if found:
                return True, final_map
            del mapping[idx1]
            used2.remove(idx2)
        return False, None

    return backtrack({}, set())


def write_atom_map(mapping, filename):
    with open(filename, "w") as f:
        for a, b in mapping.items():
            f.write(f"{a} {b}\n")


def load_atom_map(filename):
    mapping = {}
    with open(filename) as f:
        for line in f:
            if line.strip() == "":
                continue
            a, b = map(int, line.split())
            mapping[a] = b
    return mapping


# --- Kabsch alignment and robust ligand alignment ---
def kabsch_align(coords_A, coords_B):
    centroid_A = np.mean(coords_A, axis=0)
    centroid_B = np.mean(coords_B, axis=0)
    centered_A = coords_A - centroid_A
    centered_B = coords_B - centroid_B
    H = centered_B.T @ centered_A
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Vt[-1, :] *= d
    R = Vt.T @ U.T
    aligned_coords_B = (centered_B @ R) + centroid_A
    return aligned_coords_B, R, centroid_A - centroid_B


def align_ligands_with_mapping(ligandA_mol2, ligandB_mol2, aligned_ligandB_mol2, mapping):
    if not mapping:
        raise RuntimeError("No atom mapping found for alignment.")
    coordsA, namesA = parse_mol2_coords(ligandA_mol2)
    coordsB, namesB = parse_mol2_coords(ligandB_mol2)
    # Extract coordinates for mapped atoms only
    mapped_coords_A = []
    mapped_coords_B = []
    for idxA, idxB in mapping.items():
        if idxA in coordsA and idxB in coordsB:
            mapped_coords_A.append(coordsA[idxA])
            mapped_coords_B.append(coordsB[idxB])
    if len(mapped_coords_A) < 3:
        print("Not enough mapped atoms (need at least 3) for alignment.")
        return None
    coords_A = np.array(mapped_coords_A)
    coords_B = np.array(mapped_coords_B)
    aligned_coords_B, rotation_matrix, translation = kabsch_align(coords_A, coords_B)
    # Read original mol2 file to preserve all sections
    with open(ligandB_mol2, 'r') as f:
        lines = f.readlines()
    new_lines = []
    in_atoms_section = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms_section = True
            new_lines.append(line)
            continue
        elif in_atoms_section and line.startswith("@<TRIPOS>"):
            in_atoms_section = False
            new_lines.append(line)
            continue
        elif in_atoms_section:
            parts = line.split()
            if len(parts) >= 6:
                atom_id = int(parts[0])
                orig_coord = coordsB[atom_id]
                centered_coord = np.array(orig_coord) - np.mean(coords_B, axis=0)
                rotated_coord = centered_coord @ rotation_matrix
                final_coord = rotated_coord + np.mean(coords_A, axis=0)
                x, y, z = final_coord
                new_line = f"{parts[0]:>7} {parts[1]:<6} {x:>9.4f} {y:>9.4f} {z:>9.4f} {parts[5]:<6}"
                if len(parts) > 6:
                    new_line += f" {parts[6]}"
                new_line += "\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    with open(aligned_ligandB_mol2, 'w') as f:
        f.writelines(new_lines)
    print(f"Aligned ligand B to ligand A and saved to {aligned_ligandB_mol2}")
    print(f"Rotation matrix:\n{rotation_matrix}")
    print(f"Translation vector: {translation}")
    return aligned_ligandB_mol2


def align_ligandB_pdb(ligA_pdb, ligB_pdb, atom_map_file, aligned_ligB_pdb):
    mapping = load_atom_map(atom_map_file)
    if not mapping:
        raise RuntimeError("No atom mapping found for PDB alignment.")
    coordsA, namesA, atom_linesA, _ = parse_pdb_coords(ligA_pdb)
    coordsB, namesB, atom_linesB, _ = parse_pdb_coords(ligB_pdb)
    # Extract coordinates for mapped atoms
    mapped_coords_A = []
    mapped_coords_B = []
    for idxA, idxB in mapping.items():
        if idxA in coordsA and idxB in coordsB:
            mapped_coords_A.append(coordsA[idxA])
            mapped_coords_B.append(coordsB[idxB])
    if len(mapped_coords_A) < 3:
        print("Not enough mapped atoms (need at least 3) for alignment.")
        return None
    coords_A = np.array(mapped_coords_A)
    coords_B = np.array(mapped_coords_B)
    aligned_coords_B, rotation_matrix, translation = kabsch_align(coords_A, coords_B)
    # Apply transform to all atoms in ligandB.pdb
    allB_indices = sorted(coordsB.keys())
    allB_coords = np.array([coordsB[i] for i in allB_indices])
    centered_allB = allB_coords - np.mean(coords_B, axis=0)
    transformed_allB = (centered_allB @ rotation_matrix) + np.mean(coords_A, axis=0)
    write_aligned_pdb(atom_linesB, transformed_allB, aligned_ligB_pdb)
    print(f"Aligned ligandB.pdb to ligandA.pdb and saved to {aligned_ligB_pdb}")
    return aligned_ligB_pdb


def align_ligandB_gro(ligA_gro, ligB_gro, atom_map_file, aligned_ligB_gro):
    mapping = load_atom_map(atom_map_file)
    if not mapping:
        raise RuntimeError("No atom mapping found for GRO alignment.")
    coordsA, namesA, atom_linesA, _ = parse_gro_coords(ligA_gro)
    coordsB, namesB, atom_linesB, _ = parse_gro_coords(ligB_gro)
    # Extract coordinates for mapped atoms
    mapped_coords_A = []
    mapped_coords_B = []
    for idxA, idxB in mapping.items():
        if idxA in coordsA and idxB in coordsB:
            mapped_coords_A.append(coordsA[idxA])
            mapped_coords_B.append(coordsB[idxB])
    if len(mapped_coords_A) < 3:
        print("Not enough mapped atoms (need at least 3) for alignment.")
        return None
    coords_A = np.array(mapped_coords_A)
    coords_B = np.array(mapped_coords_B)
    aligned_coords_B, rotation_matrix, translation = kabsch_align(coords_A, coords_B)
    # Apply transform to all atoms in ligandB.gro
    allB_indices = sorted(coordsB.keys())
    allB_coords = np.array([coordsB[i] for i in allB_indices])
    centered_allB = allB_coords - np.mean(coords_B, axis=0)
    transformed_allB = (centered_allB @ rotation_matrix) + np.mean(coords_A, axis=0)
    write_aligned_gro(atom_linesB, transformed_allB, aligned_ligB_gro)
    print(f"Aligned ligandB.gro to ligandA.gro and saved to {aligned_ligB_gro}")
    return aligned_ligB_gro


def organize_files(args, out_dir, aligned_ligB_pdb, aligned_ligB_gro, hybrid_files=None):
    """
    Organize files into new directory structure:
    - ligand_only/
      - A_to_B/
        - lambda_0.00/ (hybrid_stateA.gro, hybrid.itp)
        - lambda_0.05/ (hybrid_stateA.gro, hybrid.itp)
        - ...
        - lambda_1.00/ (hybrid_stateA.gro, hybrid.itp)
      - B_to_A/
        - lambda_0.00/ (hybrid_stateB.gro, hybrid.itp)
        - lambda_0.05/ (hybrid_stateB.gro, hybrid.itp)
        - ...
        - lambda_1.00/ (hybrid_stateB.gro, hybrid.itp)
    - protein_complex/
      - A_to_B/
        - lambda_0.00/ (hybrid_stateA.pdb, protein.pdb, hybrid.itp)
        - lambda_0.05/ (hybrid_stateA.pdb, protein.pdb, hybrid.itp)
        - ...
        - lambda_1.00/ (hybrid_stateA.pdb, protein.pdb, hybrid.itp)
      - B_to_A/
        - lambda_0.00/ (hybrid_stateB.pdb, protein.pdb, hybrid.itp)
        - lambda_0.05/ (hybrid_stateB.pdb, protein.pdb, hybrid.itp)
        - ...
        - lambda_1.00/ (hybrid_stateB.pdb, protein.pdb, hybrid.itp)
    """
    # Create main directories
    ligand_only_dir = os.path.join(out_dir, 'ligand_only')
    protein_complex_dir = os.path.join(out_dir, 'protein_complex')

    # Create subdirectories
    ligand_a_to_b = os.path.join(ligand_only_dir, 'A_to_B')
    ligand_b_to_a = os.path.join(ligand_only_dir, 'B_to_A')
    protein_a_to_b = os.path.join(protein_complex_dir, 'A_to_B')
    protein_b_to_a = os.path.join(protein_complex_dir, 'B_to_A')

    # Generate lambda values from 0.00 to 1.00 in increments of 0.05
    lambda_values = [f"lambda_{i * 0.05:.2f}" for i in range(21)]  # 0.00 to 1.00

    # Create all lambda subdirectories
    for base_dir in [ligand_a_to_b, ligand_b_to_a, protein_a_to_b, protein_b_to_a]:
        os.makedirs(base_dir, exist_ok=True)
        for lambda_val in lambda_values:
            lambda_dir = os.path.join(base_dir, lambda_val)
            os.makedirs(lambda_dir, exist_ok=True)

    # Copy hybrid files if available
    if hybrid_files:
        hybrid_itp, hybrid_pdbA, hybrid_pdbB = hybrid_files

        # Convert PDB to GRO for ligand-only directories
        def pdb_to_gro(pdb_file, gro_file):
            """Convert PDB to GRO format"""
            with open(pdb_file, 'r') as f:
                lines = f.readlines()

            # Count atoms (skip REMARK and END lines)
            atom_lines = [line for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
            num_atoms = len(atom_lines)

            with open(gro_file, 'w') as f:
                # Write header (title line)
                f.write("Hybrid structure for FEP\n")
                # Write number of atoms
                f.write(f"{num_atoms:>5}\n")

                for i, line in enumerate(atom_lines, 1):
                    # Extract coordinates from PDB format (in Angstroms)
                    x_angstroms = float(line[30:38])
                    y_angstroms = float(line[38:46])
                    z_angstroms = float(line[46:54])
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()

                    # Convert from Angstroms to nanometers (1 nm = 10 Å)
                    x_nm = x_angstroms / 10.0
                    y_nm = y_angstroms / 10.0
                    z_nm = z_angstroms / 10.0

                    # Write GRO format line: resnum(5) resname(5) atomname(5) atomnum(5) x(8) y(8) z(8)
                    # Format: resnum(5) resname(5) atomname(5) atomnum(5) x(8.3f) y(8.3f) z(8.3f)
                    f.write(f"{1:>5}{res_name:>5}{atom_name:>5}{i:>5}{x_nm:>8.3f}{y_nm:>8.3f}{z_nm:>8.3f}\n")

                # Add box vectors (default 10nm cubic box)
                f.write("   10.00000   10.00000   10.00000\n")

        # Create files in main directories first
        hybrid_groA = os.path.join(ligand_a_to_b, 'hybrid_stateA.gro')
        hybrid_groB = os.path.join(ligand_b_to_a, 'hybrid_stateB.gro')

        pdb_to_gro(hybrid_pdbA, hybrid_groA)
        pdb_to_gro(hybrid_pdbB, hybrid_groB)

        copyfile(hybrid_itp, os.path.join(ligand_a_to_b, 'hybrid.itp'))
        copyfile(hybrid_itp, os.path.join(ligand_b_to_a, 'hybrid.itp'))

        copyfile(hybrid_pdbA, os.path.join(protein_a_to_b, 'hybrid_stateA.pdb'))
        copyfile(hybrid_pdbB, os.path.join(protein_b_to_a, 'hybrid_stateB.pdb'))
        copyfile(hybrid_itp, os.path.join(protein_a_to_b, 'hybrid.itp'))
        copyfile(hybrid_itp, os.path.join(protein_b_to_a, 'hybrid.itp'))

        # Copy protein.pdb if it exists
        if os.path.exists("protein.pdb"):
            copyfile("protein.pdb", os.path.join(protein_a_to_b, 'protein.pdb'))
            copyfile("protein.pdb", os.path.join(protein_b_to_a, 'protein.pdb'))

        # Define topol.top template content
        topol_template = """; Hybrid Ligand-Solvent Simulation Topology File

; Include forcefield parameters
#include "../../../amber14sb.ff/forcefield.itp"
; Include hybrid ligand topology
#include "hybrid.itp"
; Include water topology
#include "../../../amber14sb.ff/spce.itp"
; Include topology for ions
#include "../../../amber14sb.ff/ions.itp"


[ system ]
 Hybrid Ligand-Solvent System

[ molecules ]
; Compound        nmols
 LIG          1
"""

        # Write topol.top files to A_to_B and B_to_A level
        with open(os.path.join(ligand_a_to_b, 'topol.top'), 'w') as f:
            f.write(topol_template)
        with open(os.path.join(ligand_b_to_a, 'topol.top'), 'w') as f:
            f.write(topol_template)

    print("Output written to:")
    print(f"  {ligand_only_dir}/")
    print(f"    A_to_B/ - hybrid_stateA.gro, hybrid.itp, topol.top")
    print(f"      lambda_0.00/ to lambda_1.00/ (21 directories)")
    print(f"    B_to_A/ - hybrid_stateB.gro, hybrid.itp, topol.top")
    print(f"      lambda_0.00/ to lambda_1.00/ (21 directories)")
    print(f"  {protein_complex_dir}/")
    print(f"    A_to_B/ - hybrid_stateA.pdb, protein.pdb, hybrid.itp")
    print(f"      lambda_0.00/ to lambda_1.00/ (21 directories)")
    print(f"    B_to_A/ - hybrid_stateB.pdb, protein.pdb, hybrid.itp")
    print(f"      lambda_0.00/ to lambda_1.00/ (21 directories)")
    print(f"Total: 84 lambda directories created (21 per transition type)")
    print(f"Note: Files are placed at A_to_B and B_to_A level for building, then copied to lambda directories")


# --- Hybrid topology creation (inspired by make_hybrid.py) ---
def create_hybrid_topology(ligA_mol2, ligB_aligned_mol2, ligA_itp, ligB_itp, atom_map_file, out_itp, out_pdbA,
                           out_pdbB):
    """
    Create hybrid topology between ligandA and aligned ligandB using our custom MCS and alignment.

    This function creates a single topology that can represent both ligands through:
    - Mapped atoms: atoms that exist in both ligands (from MCS)
    - Dummy atoms: atoms that exist in only one ligand

    Args:
        ligA_mol2: Path to ligand A mol2 file
        ligB_aligned_mol2: Path to aligned ligand B mol2 file (already aligned to ligand A)
        ligA_itp: Path to ligand A itp file
        ligB_itp: Path to ligand B itp file
        atom_map_file: Path to atom mapping file
        out_itp: Output hybrid topology file
        out_pdbA: Output PDB for state A
        out_pdbB: Output PDB for state B
    """
    print("Creating hybrid topology...")

    # Load atom mapping
    mapping = load_atom_map(atom_map_file)
    if not mapping:
        raise RuntimeError("No atom mapping found for hybrid topology creation.")

    # Parse mol2 files for coordinates and atom info
    coordsA, namesA = parse_mol2_coords(ligA_mol2)
    coordsB_aligned, namesB = parse_mol2_coords(ligB_aligned_mol2)  # Use aligned coordinates

    # Parse ITP files for forcefield parameters
    atomsA = parse_itp_atoms(ligA_itp)
    atomsB = parse_itp_atoms(ligB_itp)

    # Create hybrid atoms
    hybrid_atoms = []
    atom_counter = 1

    # Process mapped atoms (exist in both ligands)
    for idxA, idxB in mapping.items():
        if idxA in atomsA and idxB in atomsB:
            atomA = atomsA[idxA]
            atomB = atomsB[idxB]

            hybrid_atom = {
                'index': atom_counter,
                'name': atomA['name'],
                'typeA': atomA['type'],
                'typeB': atomB['type'],
                'chargeA': atomA['charge'],
                'chargeB': atomB['charge'],
                'massA': atomA['mass'],
                'massB': atomB['mass'],
                'mapped': True,
                'origA_idx': idxA,
                'origB_idx': idxB
            }
            hybrid_atoms.append(hybrid_atom)
            atom_counter += 1

    # Process unique A atoms (exist only in ligand A)
    for idxA, atomA in atomsA.items():
        if idxA not in mapping:
            hybrid_atom = {
                'index': atom_counter,
                'name': 'D' + atomA['name'],  # Dummy prefix
                'typeA': atomA['type'],
                'typeB': 'DUM',  # Dummy type for B state
                'chargeA': atomA['charge'],
                'chargeB': 0.0,  # No charge in B state
                'massA': atomA['mass'],
                'massB': 0.001,  # Minimal mass in B state
                'mapped': False,
                'origA_idx': idxA,
                'origB_idx': None
            }
            hybrid_atoms.append(hybrid_atom)
            atom_counter += 1

    # Process unique B atoms (exist only in ligand B)
    for idxB, atomB in atomsB.items():
        if idxB not in [v for v in mapping.values()]:
            hybrid_atom = {
                'index': atom_counter,
                'name': 'D' + atomB['name'],  # Dummy prefix
                'typeA': 'DUM',  # Dummy type for A state
                'typeB': atomB['type'],
                'chargeA': 0.0,  # No charge in A state
                'chargeB': atomB['charge'],
                'massA': 0.001,  # Minimal mass in A state
                'massB': atomB['mass'],
                'mapped': False,
                'origA_idx': None,
                'origB_idx': idxB
            }
            hybrid_atoms.append(hybrid_atom)
            atom_counter += 1

    # Create hybrid bonds
    hybrid_bonds = create_hybrid_bonds(atomsA, atomsB, mapping, hybrid_atoms)

    # Create hybrid angles
    hybrid_angles = create_hybrid_angles(atomsA, atomsB, mapping, hybrid_atoms)

    # Create hybrid dihedrals
    hybrid_dihedrals = create_hybrid_dihedrals(atomsA, atomsB, mapping, hybrid_atoms)

    # Write hybrid topology
    write_hybrid_itp(out_itp, hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals)

    # Write hybrid PDB files for both states
    write_hybrid_pdb(out_pdbA, hybrid_atoms, coordsA, coordsB_aligned, mapping, state='A')
    write_hybrid_pdb(out_pdbB, hybrid_atoms, coordsA, coordsB_aligned, mapping, state='B')

    print(f"Created hybrid topology with {len(hybrid_atoms)} atoms")
    print(f"  - {len([a for a in hybrid_atoms if a['mapped']])} mapped atoms")
    print(f"  - {len([a for a in hybrid_atoms if not a['mapped'] and a['origA_idx']])} unique A atoms")
    print(f"  - {len([a for a in hybrid_atoms if not a['mapped'] and a['origB_idx']])} unique B atoms")


def parse_itp_atoms(itp_file):
    """Parse atom section from ITP file."""
    atoms = {}
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_atoms = False
    for line in lines:
        if line.strip() == '[ atoms ]':
            in_atoms = True
            continue
        elif in_atoms and line.strip().startswith('['):
            break
        elif in_atoms and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 8:
                atom_idx = int(parts[0])
                atom_type = parts[1]
                atom_name = parts[4]
                charge = float(parts[6])
                mass = float(parts[7])
                atoms[atom_idx] = {
                    'name': atom_name,
                    'type': atom_type,
                    'charge': charge,
                    'mass': mass
                }

    return atoms


def create_hybrid_bonds(atomsA, atomsB, mapping, hybrid_atoms):
    """Create hybrid bonds between mapped and unique atoms."""
    # This is a simplified version - in practice you'd need to parse bond sections from ITP files
    # For now, we'll create bonds based on atom proximity and mapping
    hybrid_bonds = []

    # Create bonds for mapped atoms
    for atom in hybrid_atoms:
        if atom['mapped']:
            # Find bonds involving this atom in both states
            # This would require parsing bond sections from ITP files
            pass

    return hybrid_bonds


def create_hybrid_angles(atomsA, atomsB, mapping, hybrid_atoms):
    """Create hybrid angles."""
    # Simplified - would need to parse angle sections from ITP files
    return []


def create_hybrid_dihedrals(atomsA, atomsB, mapping, hybrid_atoms):
    """Create hybrid dihedrals."""
    # Simplified - would need to parse dihedral sections from ITP files
    return []


def write_hybrid_itp(out_file, hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals):
    """Write hybrid topology file."""
    with open(out_file, 'w') as f:
        f.write("; Hybrid topology for FEP\n")
        f.write("[ moleculetype ]\n")
        f.write("; Name            nrexcl\n")
        f.write("LIG              3\n\n")

        f.write("[ atoms ]\n")
        f.write("; nr type resnr residue atom cgnr chargeA massA typeB chargeB massB\n")
        for atom in hybrid_atoms:
            f.write(f"{atom['index']:4d} {atom['typeA']:6s} {1:4d} {'LIG':6s} {atom['name']:4s} {1:4d} "
                    f"{atom['chargeA']:8.4f} {atom['massA']:7.3f} {atom['typeB']:6s} "
                    f"{atom['chargeB']:8.4f} {atom['massB']:7.3f}\n")
        f.write("\n")

        # Add bond, angle, dihedral sections if available
        if hybrid_bonds:
            f.write("[ bonds ]\n")
            f.write("; ai    aj funct\n")
            for bond in hybrid_bonds:
                f.write(f"{bond['ai']:5d} {bond['aj']:5d} {bond['funct']:5d}\n")
            f.write("\n")

        if hybrid_angles:
            f.write("[ angles ]\n")
            f.write("; ai    aj    ak funct\n")
            for angle in hybrid_angles:
                f.write(f"{angle['ai']:5d} {angle['aj']:5d} {angle['ak']:5d} {angle['funct']:5d}\n")
            f.write("\n")

        if hybrid_dihedrals:
            f.write("[ dihedrals ]\n")
            f.write("; ai    aj    ak    al funct\n")
            for dih in hybrid_dihedrals:
                f.write(f"{dih['ai']:5d} {dih['aj']:5d} {dih['ak']:5d} {dih['al']:5d} {dih['funct']:5d}\n")
            f.write("\n")


def write_hybrid_pdb(out_file, hybrid_atoms, coordsA, coordsB_aligned, mapping, state='A'):
    """Write hybrid PDB file for specified state."""
    with open(out_file, 'w') as f:
        f.write("REMARK Hybrid structure for FEP\n")

        # Calculate centroid of MCS atoms for better dummy placement
        mcs_coords = []
        for atom in hybrid_atoms:
            if atom['mapped']:
                if state == 'A' and atom['origA_idx'] in coordsA:
                    mcs_coords.append(coordsA[atom['origA_idx']])
                elif state == 'B' and atom['origB_idx'] in coordsB_aligned:
                    mcs_coords.append(coordsB_aligned[atom['origB_idx']])

        if mcs_coords:
            centroid = tuple(sum(c[i] for c in mcs_coords) / len(mcs_coords) for i in range(3))
        else:
            centroid = (0.0, 0.0, 0.0)

        for atom in hybrid_atoms:
            if state == 'A':
                if atom['origA_idx'] and atom['origA_idx'] in coordsA:
                    x, y, z = coordsA[atom['origA_idx']]
                    atom_name = atom['name'] if not atom['name'].startswith('D') else 'DUM'
                else:
                    # For unique B atoms in state A, place near MCS centroid
                    x, y, z = centroid
                    atom_name = 'DUM'
            else:  # state == 'B'
                if atom['origB_idx'] and atom['origB_idx'] in coordsB_aligned:
                    x, y, z = coordsB_aligned[atom['origB_idx']]
                    atom_name = atom['name'] if not atom['name'].startswith('D') else 'DUM'
                else:
                    # For unique A atoms in state B, place near MCS centroid
                    x, y, z = centroid
                    atom_name = 'DUM'

            f.write(f"HETATM{atom['index']:5d}  {atom_name:<4s}LIG     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

        f.write("END\n")


# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description='FEP prep: MCS, alignment, and file organization.')
    parser.add_argument('--ligA_mol2', required=True)
    parser.add_argument('--ligB_mol2', required=True)
    parser.add_argument('--ligA_pdb', required=True)
    parser.add_argument('--ligA_gro', required=True)
    parser.add_argument('--ligA_itp', required=True)
    parser.add_argument('--ligB_pdb', required=True)
    parser.add_argument('--ligB_gro', required=True)
    parser.add_argument('--ligB_itp', required=True)
    parser.add_argument('--create_hybrid', action='store_true',
                        help='Create hybrid topology for FEP simulations')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.ligA_mol2))

    # 1. Find MCS and write atom_map.txt
    gA = MolGraph.from_mol2(args.ligA_mol2)
    gB = MolGraph.from_mol2(args.ligB_mol2)
    mcs_size, mapping, atom_indicesA, atom_indicesB = find_mcs(gA, gB)
    if mcs_size < 3 or mapping is None:
        raise RuntimeError("Could not find sufficient MCS for alignment (need at least 3 atoms)")
    atom_map_file = os.path.join(out_dir, "atom_map.txt")
    write_atom_map(mapping, atom_map_file)

    # 2. Align ligandB.mol2 to ligandA.mol2 using atom_map.txt
    aligned_ligB_mol2 = os.path.join(out_dir, 'ligandB_aligned.mol2')
    align_ligands_with_mapping(args.ligA_mol2, args.ligB_mol2, aligned_ligB_mol2, mapping)

    # 3. Align ligandB.pdb to ligandA.pdb using atom_map.txt
    aligned_ligB_pdb = os.path.join(out_dir, 'ligandB_aligned.pdb')
    align_ligandB_pdb(args.ligA_pdb, args.ligB_pdb, atom_map_file, aligned_ligB_pdb)

    # 4. Align ligandB.gro to ligandA.gro using atom_map.txt
    aligned_ligB_gro = os.path.join(out_dir, 'ligandB_aligned.gro')
    align_ligandB_gro(args.ligA_gro, args.ligB_gro, atom_map_file, aligned_ligB_gro)

    # 5. Create hybrid topology if requested
    hybrid_files = None
    if args.create_hybrid:
        hybrid_itp = os.path.join(out_dir, 'hybrid.itp')
        hybrid_pdbA = os.path.join(out_dir, 'hybrid_stateA.pdb')
        hybrid_pdbB = os.path.join(out_dir, 'hybrid_stateB.pdb')
        create_hybrid_topology(args.ligA_mol2, aligned_ligB_mol2, args.ligA_itp, args.ligB_itp,
                               atom_map_file, hybrid_itp, hybrid_pdbA, hybrid_pdbB)
        hybrid_files = (hybrid_itp, hybrid_pdbA, hybrid_pdbB)

    # 6. Organize files into subdirectories
    organize_files(args, out_dir, aligned_ligB_pdb, aligned_ligB_gro, hybrid_files)


if __name__ == '__main__':
    main()
