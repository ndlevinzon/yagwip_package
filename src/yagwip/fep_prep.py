import os
import argparse
import numpy as np
from shutil import copyfile
from collections import defaultdict

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
    def from_pdb(filename):
        atoms = {}
        adj = defaultdict(set)
        idx = 1
        with open(filename) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                element = line[76:78].strip().upper() if len(line) >= 78 else line[12:14].strip().upper()
                atoms[idx] = Atom(idx, element)
                idx += 1
        # No bond info in PDB, so no bonds/adjacency
        g = MolGraph()
        g.atoms = atoms
        g.bonds = []
        g.adj = adj
        g.bond_types = {}
        return g

    def subgraph(self, atom_indices):
        sg = MolGraph()
        for idx in atom_indices:
            atom = self.atoms[idx]
            sg.atoms[idx] = Atom(idx, atom.element)
        for idx in sg.atoms:
            sg.atoms[idx].neighbors = set()
            sg.atoms[idx].degree = 0
        return sg

def are_isomorphic(g1, g2):
    if len(g1.atoms) != len(g2.atoms):
        return False, None
    candidates = {}
    for idx1, atom1 in g1.atoms.items():
        candidates[idx1] = [
            idx2
            for idx2, atom2 in g2.atoms.items()
            if atom1.element == atom2.element
        ]
        if not candidates[idx1]:
            return False, None
    def backtrack(mapping, used2):
        if len(mapping) == len(g1.atoms):
            return True, dict(mapping)
        idx1 = next(i for i in g1.atoms if i not in mapping)
        for idx2 in candidates[idx1]:
            if idx2 in used2:
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

def enumerate_connected_subgraphs(graph, size):
    # For PDBs, treat as all possible combinations of atom indices of given size
    from itertools import combinations
    return [set(s) for s in combinations(graph.atoms.keys(), size)]

def find_mcs(g1, g2):
    if len(g1.atoms) > len(g2.atoms):
        g1, g2 = g2, g1
    for size in range(len(g1.atoms), 0, -1):
        subgraphs1 = enumerate_connected_subgraphs(g1, size)
        if not subgraphs1:
            continue
        for atom_indices1 in subgraphs1:
            sg1 = g1.subgraph(atom_indices1)
            elem_count1 = defaultdict(int)
            for a in sg1.atoms.values():
                elem_count1[a.element] += 1
            subgraphs2 = [
                s
                for s in enumerate_connected_subgraphs(g2, size)
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

# --- PDB parsing and writing ---
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

# --- Kabsch alignment using MCS mapping ---
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

def align_ligandB_to_A_mcs(ligA_pdb, ligB_pdb, out_pdb):
    # Build graphs
    gA = MolGraph.from_pdb(ligA_pdb)
    gB = MolGraph.from_pdb(ligB_pdb)
    # Find MCS
    mcs_size, mapping, atom_indicesA, atom_indicesB = find_mcs(gA, gB)
    if mcs_size < 3 or mapping is None or atom_indicesA is None or atom_indicesB is None:
        raise RuntimeError("Could not find sufficient MCS for alignment (need at least 3 atoms)")
    # Get coordinates for mapped atoms
    coordsA_all, _, atom_linesA, _ = parse_pdb_coords(ligA_pdb)
    coordsB_all, _, atom_linesB, linesB = parse_pdb_coords(ligB_pdb)
    mappedA = sorted(atom_indicesA)
    mappedB = [mapping[a] for a in mappedA]
    coordsA = np.array([coordsA_all[a] for a in mappedA])
    coordsB = np.array([coordsB_all[b] for b in mappedB])
    # Kabsch align
    aligned_coordsB, R, t = kabsch_align(coordsA, coordsB)
    # Apply transformation to all atoms in ligandB
    allB_indices = sorted(coordsB_all.keys())
    allB_coords = np.array([coordsB_all[i] for i in allB_indices])
    centroid_B = np.mean(coordsB, axis=0)
    centered_allB = allB_coords - centroid_B
    transformed_allB = (centered_allB @ R) + np.mean(coordsA, axis=0)
    # Write aligned pdb
    write_aligned_pdb(atom_linesB, transformed_allB, out_pdb)

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description='FEP prep: align ligands (pdb, MCS), generate water/complex systems.')
    parser.add_argument('--ligA_pdb', required=True)
    parser.add_argument('--ligA_itp', required=True)
    parser.add_argument('--ligB_pdb', required=True)
    parser.add_argument('--ligB_itp', required=True)
    parser.add_argument('--protein_gro', required=True)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.ligA_pdb))
    aligned_ligB_pdb = os.path.join(out_dir, 'ligandB_aligned.pdb')
    align_ligandB_to_A_mcs(args.ligA_pdb, args.ligB_pdb, aligned_ligB_pdb)

    out_dirs = ['A_water', 'A_complex', 'B_water', 'B_complex']
    out_dirs_full = [os.path.join(out_dir, d) for d in out_dirs]
    for d in out_dirs_full:
        os.makedirs(d, exist_ok=True)

    copyfile(args.ligA_pdb, os.path.join(out_dirs_full[0], 'ligandA.pdb'))
    copyfile(args.ligA_itp, os.path.join(out_dirs_full[0], 'ligandA.itp'))
    copyfile(args.ligA_pdb, os.path.join(out_dirs_full[1], 'ligandA.pdb'))
    copyfile(args.ligA_itp, os.path.join(out_dirs_full[1], 'ligandA.itp'))
    copyfile(aligned_ligB_pdb, os.path.join(out_dirs_full[2], 'ligandB.pdb'))
    copyfile(args.ligB_itp, os.path.join(out_dirs_full[2], 'ligandB.itp'))
    copyfile(aligned_ligB_pdb, os.path.join(out_dirs_full[3], 'ligandB.pdb'))
    copyfile(args.ligB_itp, os.path.join(out_dirs_full[3], 'ligandB.itp'))
    copyfile(args.protein_gro, os.path.join(out_dirs_full[1], 'protein.gro'))
    copyfile(args.protein_gro, os.path.join(out_dirs_full[3], 'protein.gro'))

    print('TODO: Run pdb2gmx, solvate, genion for each system using YAGWIP utilities.')
    print("Output written to:")
    print(f"  {aligned_ligB_pdb}")
    for d in out_dirs_full:
        print(f"  {d}/")
        for f in os.listdir(d):
            print(f"    {f}")

if __name__ == '__main__':
    main()
