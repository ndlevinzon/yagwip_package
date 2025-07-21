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

def are_isomorphic(g1, g2):
    if len(g1.atoms) != len(g2.atoms):
        return False, None
    candidates = {}
    for idx1, atom1 in g1.atoms.items():
        candidates[idx1] = [
            idx2
            for idx2, atom2 in g2.atoms.items()
            if atom1.element == atom2.element and atom1.degree == atom2.degree
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

# --- MOL2 and PDB parsing ---
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

# --- Kabsch alignment using MCS mapping from mol2, applied to pdbs ---
def align_ligandB_to_A_mcs_mol2pdb(ligA_mol2, ligB_mol2, ligA_pdb, ligB_pdb, out_pdb):
    # 1. Find MCS using mol2s
    gA = MolGraph.from_mol2(ligA_mol2)
    gB = MolGraph.from_mol2(ligB_mol2)
    mcs_size, mapping, atom_indicesA, atom_indicesB = find_mcs(gA, gB)
    if mcs_size < 3 or mapping is None or atom_indicesA is None or atom_indicesB is None:
        raise RuntimeError("Could not find sufficient MCS for alignment (need at least 3 atoms)")
    # 2. Parse mol2 and pdb files
    mol2A_coords, mol2A_names = parse_mol2_coords(ligA_mol2)
    mol2B_coords, mol2B_names = parse_mol2_coords(ligB_mol2)
    pdbA_coords, pdbA_names, pdbA_lines, _ = parse_pdb_coords(ligA_pdb)
    pdbB_coords, pdbB_names, pdbB_lines, _ = parse_pdb_coords(ligB_pdb)
    # 3. Build mapping from mol2 idx to pdb idx (assume order, fallback to name)
    def mol2_to_pdb_idx(mol2_names, pdb_names):
        mapping = {}
        used = set()
        for m_idx, m_name in mol2_names.items():
            # Try to find the first pdb idx with the same name not already used
            for p_idx, p_name in pdb_names.items():
                if p_name == m_name and p_idx not in used:
                    mapping[m_idx] = p_idx
                    used.add(p_idx)
                    break
            else:
                # fallback: use same index if possible
                if m_idx in pdb_names and m_name == pdb_names[m_idx]:
                    mapping[m_idx] = m_idx
        return mapping
    mapA = mol2_to_pdb_idx(mol2A_names, pdbA_names)
    mapB = mol2_to_pdb_idx(mol2B_names, pdbB_names)
    # 4. Extract coordinates for mapped atoms from PDBs using MCS mapping
    mappedA = sorted(atom_indicesA)
    mappedB = [mapping[a] for a in mappedA]
    pdbA_indices = [mapA[a] for a in mappedA if a in mapA]
    pdbB_indices = [mapB[mapping[a]] for a in mappedA if a in mapA and mapping[a] in mapB]
    if len(pdbA_indices) < 3 or len(pdbB_indices) < 3:
        raise RuntimeError("Could not find enough mapped atoms in PDBs for alignment.")
    coordsA = np.array([pdbA_coords[i] for i in pdbA_indices])
    coordsB = np.array([pdbB_coords[i] for i in pdbB_indices])
    # 5. Kabsch align
    centroid_A = np.mean(coordsA, axis=0)
    centroid_B = np.mean(coordsB, axis=0)
    centered_A = coordsA - centroid_A
    centered_B = coordsB - centroid_B
    H = centered_B.T @ centered_A
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Vt[-1, :] *= d
    R = Vt.T @ U.T
    # Apply transformation to all atoms in ligandB.pdb
    allB_indices = sorted(pdbB_coords.keys())
    allB_coords = np.array([pdbB_coords[i] for i in allB_indices])
    centered_allB = allB_coords - centroid_B
    transformed_allB = (centered_allB @ R) + centroid_A
    # Write aligned pdb
    write_aligned_pdb(pdbB_lines, transformed_allB, out_pdb)

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description='FEP prep: align ligands (mol2 MCS, pdb alignment), generate water/complex systems.')
    parser.add_argument('--ligA_pdb', required=True)
    parser.add_argument('--ligA_itp', required=True)
    parser.add_argument('--ligB_pdb', required=True)
    parser.add_argument('--ligB_itp', required=True)
    parser.add_argument('--protein_pdb', required=True)
    parser.add_argument('--ligA_mol2', required=False)
    parser.add_argument('--ligB_mol2', required=False)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.ligA_pdb))
    ligA_mol2 = args.ligA_mol2 or os.path.join(out_dir, 'ligandA.mol2')
    ligB_mol2 = args.ligB_mol2 or os.path.join(out_dir, 'ligandB.mol2')
    aligned_ligB_pdb = os.path.join(out_dir, 'ligandB_aligned.pdb')
    align_ligandB_to_A_mcs_mol2pdb(ligA_mol2, ligB_mol2, args.ligA_pdb, args.ligB_pdb, aligned_ligB_pdb)

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
    copyfile(args.protein_pdb, os.path.join(out_dirs_full[1], 'protein.pdb'))
    copyfile(args.protein_pdb, os.path.join(out_dirs_full[3], 'protein.pdb'))

    print('TODO: Run pdb2gmx, solvate, genion for each system using YAGWIP utilities.')
    print("Output written to:")
    print(f"  {aligned_ligB_pdb}")
    for d in out_dirs_full:
        print(f"  {d}/")
        for f in os.listdir(d):
            print(f"    {f}")

if __name__ == '__main__':
    main()
