import os
import argparse
import numpy as np
from shutil import copyfile
from collections import defaultdict

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

def find_mcs(g1, g2):
    # Legacy approach: try to match all atoms by element and degree, using backtracking
    # Returns the largest mapping found
    best_size = 0
    best_mapping = None
    best_atoms1 = None
    best_atoms2 = None
    nA = len(g1.atoms)
    nB = len(g2.atoms)
    # Try all possible sizes from min(nA, nB) down to 3
    for size in range(min(nA, nB), 2, -1):
        # Try all combinations of size atoms from g1
        from itertools import combinations
        for atoms1 in combinations(g1.atoms.keys(), size):
            sg1 = g1.subgraph(atoms1)
            # Try all combinations of size atoms from g2
            for atoms2 in combinations(g2.atoms.keys(), size):
                sg2 = g2.subgraph(atoms2)
                iso, mapping = are_isomorphic(sg1, sg2)
                if iso:
                    # mapping: sg1 idx -> sg2 idx
                    # Convert mapping to original indices
                    orig_mapping = {list(atoms1)[i]: list(atoms2)[j] for i, j in mapping.items()}
                    if size > best_size:
                        best_size = size
                        best_mapping = orig_mapping
                        best_atoms1 = set(atoms1)
                        best_atoms2 = set(atoms2)
                    break  # Only need one mapping of this size
            if best_size == size:
                break
        if best_size == size:
            break
    if best_size == 0:
        return 0, None, None, None
    return best_size, best_mapping, best_atoms1, best_atoms2

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

def organize_files(args, out_dir, aligned_ligB_pdb):
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
    for d in out_dirs_full:
        print(f"  {d}/")
        for f in os.listdir(d):
            print(f"    {f}")

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description='FEP prep: MCS, alignment, and file organization.')
    parser.add_argument('--ligA_mol2', required=True)
    parser.add_argument('--ligB_mol2', required=True)
    parser.add_argument('--ligA_pdb', required=True)
    parser.add_argument('--ligA_itp', required=True)
    parser.add_argument('--ligB_pdb', required=True)
    parser.add_argument('--ligB_itp', required=True)
    parser.add_argument('--protein_pdb', required=True)
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

    # 4. Organize files into subdirectories
    organize_files(args, out_dir, aligned_ligB_pdb)

if __name__ == '__main__':
    main()
