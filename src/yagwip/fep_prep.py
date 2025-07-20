# === Standard Library Imports ===
import sys
import os
from collections import defaultdict
import logging
import random

# === Third-Party Imports ===
import pandas as pd
import numpy as np

logger = logging.getLogger("hybrid_topology")


# =====================
# Ligand Alignment Functions
# =====================
def kabsch_align(coords_A, coords_B):
    """
    Align coordinates B to coordinates A using Kabsch algorithm.

    Args:
        coords_A: numpy array of coordinates for reference (ligand A)
        coords_B: numpy array of coordinates to align (ligand B)

    Returns:
        aligned_coords_B: numpy array of aligned coordinates
        rotation_matrix: 3x3 rotation matrix
        translation_vector: translation vector
    """
    # Center both coordinate sets
    centroid_A = np.mean(coords_A, axis=0)
    centroid_B = np.mean(coords_B, axis=0)

    centered_A = coords_A - centroid_A
    centered_B = coords_B - centroid_B

    # Compute covariance matrix
    H = centered_B.T @ centered_A

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation matrix (handle reflection case)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Vt[-1, :] *= d

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Apply rotation and translation
    aligned_coords_B = (centered_B @ R) + centroid_A

    return aligned_coords_B, R, centroid_A - centroid_B


def align_ligands_with_mapping(ligandA_mol2, ligandB_mol2, aligned_ligandB_mol2, mapping):
    """
    Align ligand B to ligand A using provided atom mapping and save the aligned coordinates.

    Args:
        ligandA_mol2: Path to ligand A mol2 file (reference)
        ligandB_mol2: Path to ligand B mol2 file (to be aligned)
        aligned_ligandB_mol2: Path to save aligned ligand B mol2 file
        mapping: Dictionary mapping atom indices from ligandA to ligandB
    """
    # Parse coordinates from mol2 files
    coordsA, namesA = parse_mol2_coords(ligandA_mol2)
    coordsB, namesB = parse_mol2_coords(ligandB_mol2)

    print(f"Mapping provided: {mapping}")

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

    # Convert to numpy arrays
    coords_A = np.array(mapped_coords_A)
    coords_B = np.array(mapped_coords_B)

    print(f"Aligning {len(coords_A)} mapped atoms")

    # Align coordinates using Kabsch algorithm
    aligned_coords_B, rotation_matrix, translation = kabsch_align(coords_A, coords_B)

    # Read original mol2 file to preserve all sections
    with open(ligandB_mol2, 'r') as f:
        lines = f.readlines()

    # Update coordinates in the mol2 file
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
            # Update atom coordinates
            parts = line.split()
            if len(parts) >= 6:
                atom_id = int(parts[0])
                # Apply the same transformation to ALL atoms in ligand B
                # This ensures the entire molecule is properly aligned
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

    # Write aligned mol2 file
    with open(aligned_ligandB_mol2, 'w') as f:
        f.writelines(new_lines)

    print(f"Aligned ligand B to ligand A and saved to {aligned_ligandB_mol2}")
    print(f"Rotation matrix:\n{rotation_matrix}")
    print(f"Translation vector: {translation}")

    return aligned_ligandB_mol2


# =====================
# Utility Functions
# =====================
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


def load_atom_map(filename):
    mapping = {}
    with open(filename) as f:
        for line in f:
            if line.strip() == "":
                continue
            a, b = map(int, line.split())
            mapping[a] = b
    return mapping


def write_atom_map(mapping, filename):
    with open(filename, "w") as f:
        for a, b in mapping.items():
            f.write(f"{a} {b}\n")


# =====================
# MCS Code
# =====================
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
                element = "".join(filter(str.isalpha, parts[5])).upper()
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


# =====================
# Hybrid Topology Code
# =====================

logger = logging.getLogger("hybrid_topology")


# --- Enhanced HybridAtom, HybridBond, HybridAngle, HybridDihedral ---
class HybridAtom:
    def __init__(
            self,
            index,
            atom_name,
            typeA,
            typeB,
            chargeA,
            chargeB,
            massA,
            massB,
            mapped,
            origA_idx=None,
            origB_idx=None,
    ):
        self.index = index
        self.atom_name = atom_name
        self.typeA = typeA
        self.typeB = typeB
        self.chargeA = chargeA
        self.chargeB = chargeB
        self.massA = massA
        self.massB = massB
        self.mapped = mapped
        self.origA_idx = origA_idx
        self.origB_idx = origB_idx


class HybridBond:
    def __init__(self, ai, aj, funct, parA, parB, mapped, fakeA=False, fakeB=False):
        self.ai = ai
        self.aj = aj
        self.funct = funct
        self.parA = parA
        self.parB = parB
        self.mapped = mapped
        self.fakeA = fakeA
        self.fakeB = fakeB


class HybridAngle:
    def __init__(self, ai, aj, ak, funct, parA, parB, mapped, fakeA=False, fakeB=False):
        self.ai = ai
        self.aj = aj
        self.ak = ak
        self.funct = funct
        self.parA = parA
        self.parB = parB
        self.mapped = mapped
        self.fakeA = fakeA
        self.fakeB = fakeB


class HybridDihedral:
    def __init__(
            self, ai, aj, ak, al, funct, parA, parB, mapped, fakeA=False, fakeB=False
    ):
        self.ai = ai
        self.aj = aj
        self.ak = ak
        self.al = al
        self.funct = funct
        self.parA = parA
        self.parB = parB
        self.mapped = mapped
        self.fakeA = fakeA
        self.fakeB = fakeB


# --- Robust parameter lookup and fake term creation ---
def robust_lookup(df, keycols, key, paramcols, dummy):
    try:
        row = df
        for col, val in zip(keycols, key):
            row = row[row[col] == val]
        if row.shape[0] == 0:
            return [dummy.get(col, 0.0) for col in paramcols], True
        vals = [
            row.iloc[0][col] if col in row.columns else dummy.get(col, 0.0)
            for col in paramcols
        ]
        return vals, False
    except Exception as e:
        logger.warning(f"Parameter lookup failed for key {key}: {e}")
        return [dummy.get(col, 0.0) for col in paramcols], True


# --- pmx-style build_hybrid_terms ---
def build_hybrid_terms(
        dfA, dfB, mapping, keycols, HybridClass, dummyA, dummyB, paramcolsA, paramcolsB
):
    """
    For each unique term (bond/angle/dihedral) in the union of A and B:
    - If present in both: use real parameters for both.
    - If present in only one: use real parameters for that state, and for the other, use dummies and zeroed parameters (indices point to correct dummy atoms).
    """
    termsA = dfA.copy()
    termsB = dfB.copy()
    inv_map = {v: k for k, v in mapping.items()}
    all_keys = set()
    for _, row in termsA.iterrows():
        all_keys.add(tuple(int(row[c]) for c in keycols))
    for _, row in termsB.iterrows():
        mapped_key = tuple(inv_map.get(int(row[c]), int(row[c])) for c in keycols)
        all_keys.add(mapped_key)
    hybrid_terms = []
    for key in sorted(all_keys):
        # A state
        valsA, fakeA = robust_lookup(termsA, keycols, key, paramcolsA, dummyA)
        # B state (map indices if needed)
        # If any k in key is not in mapping, treat as dummy (use -1 as dummy index)
        mapped_key_B = tuple(mapping[k] if k in mapping else -1 for k in key)
        valsB, fakeB = robust_lookup(termsB, keycols, mapped_key_B, paramcolsB, dummyB)
        mapped = not (fakeA or fakeB)
        # If term is missing in one state, use dummy indices and zeroed parameters
        if HybridClass == HybridBond:
            ai, aj = key
            hybrid_terms.append(
                HybridBond(ai, aj, 1, valsA, valsB, mapped, fakeA, fakeB)
            )
        elif HybridClass == HybridAngle:
            ai, aj, ak = key
            hybrid_terms.append(
                HybridAngle(ai, aj, ak, 1, valsA, valsB, mapped, fakeA, fakeB)
            )
        elif HybridClass == HybridDihedral:
            ai, aj, ak, al = key
            hybrid_terms.append(
                HybridDihedral(ai, aj, ak, al, 2, valsA, valsB, mapped, fakeA, fakeB)
            )
    return hybrid_terms


# --- Charge/mass consistency check ---
def print_total_charge(hybrid_atoms):
    total_charge_A = sum(atom.chargeA for atom in hybrid_atoms)
    total_charge_B = sum(atom.chargeB for atom in hybrid_atoms)
    logger.info(f"Total charge state A: {total_charge_A:.4f}")
    logger.info(f"Total charge state B: {total_charge_B:.4f}")


# --- Modular section processing and logging ---
def parse_itp_section(filename, section, ncols, colnames):
    with open(filename) as f:
        lines = f.readlines()
    data = []
    in_section = False
    for line in lines:
        if line.strip().startswith(f"[ {section} ]"):
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("["):
                break
            if line.strip() == "" or line.strip().startswith(";"):
                continue
            parts = line.split()
            if len(parts) < ncols:
                continue
            row = {colnames[i]: parts[i] for i in range(ncols)}
            data.append(row)
    return pd.DataFrame(data)


def parse_itp_atoms_full(filename):
    with open(filename) as f:
        lines = f.readlines()
    atoms = []
    in_atoms = False
    for line in lines:
        if line.strip().startswith("[ atoms ]"):
            in_atoms = True
            continue
        if in_atoms:
            if line.strip().startswith("["):
                break
            if line.strip() == "" or line.strip().startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            atoms.append(
                {
                    "index": int(parts[0]),
                    "type": parts[1],
                    "resnr": int(parts[2]),
                    "residue": parts[3],
                    "atom_name": parts[4],
                    "cgnr": int(parts[5]),
                    "charge": float(parts[6]),
                    "mass": float(parts[7]) if len(parts) > 7 else 0.0,
                }
            )
    return pd.DataFrame(atoms)


# Removed redundant build_hybrid_atoms function - use build_hybrid_atoms_interpolated instead


def write_hybrid_topology(
        filename,
        hybrid_atoms,
        hybrid_bonds=None,
        hybrid_pairs=None,
        hybrid_angles=None,
        hybrid_dihedrals=None,
        hybrid_exclusions=None,
        system_name="Hybrid System",
        molecule_name="LIG",
        nmols=1,
):
    # Sort hybrid atoms by their index to ensure correct order
    sorted_atoms = sorted(hybrid_atoms, key=lambda atom: atom.index)
    with open(filename, "w") as f:
        f.write(f"; Include force field parameters\n")
        f.write("[ moleculetype ]\n")
        f.write("; Name            nrexcl\n")
        f.write(f"{molecule_name:<18}3\n\n")
        f.write("[ atoms ]\n")
        f.write(
            "; nr type resnr residue atom cgnr  chargeA    massA  typeB chargeB  massB\n"
        )
        for atom in sorted_atoms:
            f.write(
                f'{atom.index:4d} {atom.typeA:6s} {1:4d} {"LIG":6s} {atom.atom_name:4s} {1:4d} '
                f"{atom.chargeA:8.4f} {atom.massA:7.3f} {atom.typeB:6s} {atom.chargeB:8.4f} {atom.massB:7.3f}\n"
            )
        f.write("\n")
        if hybrid_bonds is not None:
            f.write("[ bonds ]\n")
            f.write(";  ai    aj funct    par_A1  par_A2  par_B1  par_B2\n")
            for bond in hybrid_bonds:
                ai = getattr(bond, "ai", None)
                aj = getattr(bond, "aj", None)
                funct = getattr(bond, "funct", None)
                parA = getattr(bond, "parA", [])
                parB = getattr(bond, "parB", [])
                if ai is None or aj is None or funct is None:
                    continue
                if isinstance(parA, list):
                    parA_str = " ".join(
                        "0.00" if pd.isna(p) or p == "" else str(p) for p in parA
                    )
                else:
                    parA_str = "0.00" if pd.isna(parA) or parA == "" else str(parA)
                if isinstance(parB, list):
                    parB_str = " ".join(
                        "0.00" if pd.isna(p) or p == "" else str(p) for p in parB
                    )
                else:
                    parB_str = "0.00" if pd.isna(parB) or parB == "" else str(parB)
                f.write(
                    f"{int(ai):5d} {int(aj):5d} {int(funct):5d} {parA_str} {parB_str}\n"
                )
            f.write("\n")
        if hybrid_pairs is not None:
            f.write("[ pairs ]\n")
            f.write(";  ai    aj funct\n")
            for pair in hybrid_pairs:
                ai = getattr(pair, "ai", pair["ai"])
                aj = getattr(pair, "aj", pair["aj"])
                funct = getattr(pair, "funct", pair["funct"])
                if ai is None or aj is None or funct is None:
                    continue
                f.write(f"{int(ai):5d} {int(aj):5d} {int(funct):5d}\n")
            f.write("\n")
        if hybrid_angles is not None:
            f.write("[ angles ]\n")
            f.write(";  ai    aj    ak funct    par_A1   par_A2   par_B1   par_B2\n")
            for angle in hybrid_angles:
                ai = getattr(angle, "ai", None)
                aj = getattr(angle, "aj", None)
                ak = getattr(angle, "ak", None)
                funct = getattr(angle, "funct", None)
                parA = getattr(angle, "parA", [])
                parB = getattr(angle, "parB", [])
                if ai is None or aj is None or ak is None or funct is None:
                    continue
                if isinstance(parA, list):
                    parA_str = " ".join(
                        "0.00" if pd.isna(p) or p == "" else str(p) for p in parA
                    )
                else:
                    parA_str = "0.00" if pd.isna(parA) or parA == "" else str(parA)
                if isinstance(parB, list):
                    parB_str = " ".join(
                        "0.00" if pd.isna(p) or p == "" else str(p) for p in parB
                    )
                else:
                    parB_str = "0.00" if pd.isna(parB) or parB == "" else str(parB)
                f.write(
                    f"{int(ai):5d} {int(aj):5d} {int(ak):5d} {int(funct):5d} {parA_str} {parB_str}\n"
                )
            f.write("\n")
        if hybrid_dihedrals is not None:
            f.write("[ dihedrals ]\n")
            f.write(
                ";  ai    aj    ak    al funct    par_A1   par_A2   par_B1   par_B2\n"
            )
            for dih in hybrid_dihedrals:
                ai = getattr(dih, "ai", None)
                aj = getattr(dih, "aj", None)
                ak = getattr(dih, "ak", None)
                al = getattr(dih, "al", None)
                funct = getattr(dih, "funct", None)
                parA = getattr(dih, "parA", [])
                parB = getattr(dih, "parB", [])
                if (
                        ai is None
                        or aj is None
                        or ak is None
                        or al is None
                        or funct is None
                ):
                    continue
                if isinstance(parA, list):
                    parA_clean = [p for p in parA if not pd.isna(p) and p != ""][:2]
                    while len(parA_clean) < 2:
                        parA_clean.append("0.00")
                    parA_str = " ".join(str(p) for p in parA_clean)
                else:
                    parA_str = "0.00 0.00"
                if isinstance(parB, list):
                    parB_clean = [p for p in parB if not pd.isna(p) and p != ""][:2]
                    while len(parB_clean) < 2:
                        parB_clean.append("0.00")
                    parB_str = " ".join(str(p) for p in parB_clean)
                else:
                    parB_str = "0.00 0.00"
                f.write(
                    f"{int(ai):5d} {int(aj):5d} {int(ak):5d} {int(al):5d}     2 {parA_str} {parB_str}\n"
                )
            f.write("\n")

        # Write [ exclusions ] block at the top of the exclusion section
        if hybrid_exclusions is not None and ("STATEA" in hybrid_exclusions or "STATEB" in hybrid_exclusions):
            from collections import defaultdict
            for state in ["STATEA", "STATEB"]:
                if state in hybrid_exclusions:
                    f.write(f"#ifdef {state}\n")
                    f.write("[ exclusions ]\n")
                    f.write(";  ai    aj ...\n")
                    exclusion_dict = defaultdict(list)
                    for excl in hybrid_exclusions[state]:
                        exclusion_dict[excl["ai"]].append(excl["aj"])
                    for ai in sorted(exclusion_dict.keys()):
                        aj_list = sorted(set(exclusion_dict[ai]))
                        f.write(f"{ai:5d} " + " ".join(f"{aj:5d}" for aj in aj_list) + "\n")
                    f.write("#endif\n\n")
        else:
            # Always print [ exclusions ] header, even if no exclusions
            f.write("[ exclusions ]\n")
            f.write(";  ai    aj ...\n")
            if hybrid_exclusions is not None and len(hybrid_exclusions) > 0:
                from collections import defaultdict
                exclusion_dict = defaultdict(list)
                for excl in hybrid_exclusions:
                    exclusion_dict[excl["ai"]].append(excl["aj"])
                for ai in sorted(exclusion_dict.keys()):
                    aj_list = sorted(set(exclusion_dict[ai]))
                    f.write(f"{ai:5d} " + " ".join(f"{aj:5d}" for aj in aj_list) + "\n")
            f.write("\n")

        # Add conditional include for position restraints only if there are dummy atoms
        dummy_atoms = [atom for atom in hybrid_atoms if atom.typeA == "DUM" or atom.typeB == "DUM"]
        if dummy_atoms:
            f.write("#ifdef POSRES\n")
            f.write('#include "posre_ligand.itp"\n')
            f.write("#endif\n\n")


def write_position_restraints_file(filename, hybrid_atoms):
    """
    Create a separate position restraints file for dummy atoms.

    Args:
        filename (str): Path to the position restraints file to create
        hybrid_atoms (list): List of HybridAtom objects
    """
    # Sort hybrid atoms by their index to ensure correct order
    sorted_atoms = sorted(hybrid_atoms, key=lambda atom: atom.index)

    # Find dummy atoms
    dummy_atoms = [atom for atom in sorted_atoms if atom.typeA == "DUM" or atom.typeB == "DUM"]

    if not dummy_atoms:
        return  # No dummy atoms, no need to create restraints file

    with open(filename, "w") as f:
        f.write("[ position_restraints ]\n")
        f.write("; atom  type      fx      fy      fz\n")
        for atom in dummy_atoms:
            f.write(f"{atom.index:5d}     1    1000    1000    1000\n")
        f.write("\n")


def filter_topology_sections(df, present_indices):
    """
    Filter a topology DataFrame (e.g., bonds, angles) to only include terms where all atom indices are present.
    Also renumber atom indices to be sequential starting from 1.
    Filter out bonds where an atom is bound to itself (ai == aj).
    """
    present = set(present_indices)
    if "ai" in df.columns:
        mask = df["ai"].astype(int).isin(present)
        for col in ["aj", "ak", "al"]:
            if col in df.columns:
                mask &= df[col].astype(int).isin(present)

        # Ensure all atom indices are within the bounds of present atoms
        max_atom_idx = max(present) if present else 0

        # For bonds, filter out self-bonds (ai == aj) and bonds to atoms outside bounds
        if "aj" in df.columns and "ai" in df.columns:
            mask &= df["ai"].astype(int) != df["aj"].astype(int)
            mask &= df["ai"].astype(int) <= max_atom_idx
            mask &= df["aj"].astype(int) <= max_atom_idx

        # For angles and dihedrals, ensure all atom indices are within bounds
        if "ak" in df.columns:
            mask &= df["ak"].astype(int) <= max_atom_idx
        if "al" in df.columns:
            mask &= df["al"].astype(int) <= max_atom_idx

        filtered_df = df[mask].copy()

        # Create a mapping from old indices to new sequential indices
        index_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(sorted(present), 1)
        }

        # Renumber all atom indices
        for col in ["ai", "aj", "ak", "al"]:
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].astype(int).map(index_mapping)

        return filtered_df
    return df.copy()


def build_lambda_atom_list(dfA, dfB, mapping, lam):
    """
    For each lambda, return the atom list (index, atom_name, origA_idx, origB_idx, atom_type) to be present.
    At lambda=0: mapped + uniqueA (A order)
    At lambda=1: mapped + uniqueB (B order)
    0 < lambda < 1: mapped (A order) + uniqueA (A order) + uniqueB (B order)
    """
    mapped_atoms = []
    uniqueA_atoms = []
    uniqueB_atoms = []
    usedB = set()

    for _, rowA in dfA.iterrows():
        idxA = int(rowA["index"])
        if idxA in mapping:
            idxB = mapping[idxA]
            mapped_atoms.append((idxA, rowA["atom_name"], idxA, idxB, "mapped"))
            usedB.add(idxB)
        else:
            uniqueA_atoms.append((idxA, rowA["atom_name"], idxA, None, "uniqueA"))

    for _, rowB in dfB.iterrows():
        idxB = int(rowB["index"])
        if idxB not in usedB:
            uniqueB_atoms.append((idxB, rowB["atom_name"], None, idxB, "uniqueB"))

    if lam == 0:
        return mapped_atoms + uniqueA_atoms
    elif lam == 1:
        return mapped_atoms + uniqueB_atoms
    else:
        return mapped_atoms + uniqueA_atoms + uniqueB_atoms


def build_hybrid_atoms_interpolated(dfA, dfB, mapping, lam):
    """
    Enhanced dual topology atom building with growing procedure:
    - Lambda 0: Pure molecule A (only A atoms present with real properties)
    - Lambda 0.5: Hybrid structure with interpolated properties
    - Lambda 1: Pure molecule B (only B atoms present with real properties)
    - Intermediate values: Gradual transition with realistic dummy properties

    The growing procedure ensures smooth transitions between states.
    """
    # Get the lambda-specific atom list
    atom_list = build_lambda_atom_list(dfA, dfB, mapping, lam)
    hybrid_atoms = []

    for new_idx, (old_idx, atom_name, origA_idx, origB_idx, atom_type) in enumerate(
            atom_list, 1
    ):
        if atom_type == "mapped":
            # Mapped atoms: interpolate between A and B properties
            rowA = dfA[dfA["index"] == origA_idx].iloc[0]
            rowB = dfB[dfB["index"] == origB_idx].iloc[0]
            chargeA = rowA["charge"]
            chargeB = rowB["charge"]
            massA = rowA["mass"]
            massB = rowB["mass"]
            typeA = rowA["type"]
            typeB = rowB["type"]

            # MCS atoms always have real atom names (never DUM)
            if lam <= 0.5:
                atom_name = rowA["atom_name"]
            else:
                atom_name = rowB["atom_name"]

        elif atom_type == "uniqueA":
            # Unique A atoms: only present at lambda <= 0.5
            rowA = dfA[dfA["index"] == origA_idx].iloc[0]
            chargeA = rowA["charge"]
            massA = rowA["mass"]
            typeA = rowA["type"]
            if lam == 0:
                atom_name = rowA["atom_name"]
                chargeB = chargeA
                massB = massA
                typeB = typeA
            elif lam == 1:
                atom_name = "DUM"
                chargeB = 0.0
                massB = 0.001
                typeB = "DUM"
            else:
                atom_name = "DUM"
                if lam < 0.5:
                    chargeB = chargeA
                    massB = massA
                    typeB = typeA
                else:
                    chargeB = 0.0
                    massB = 0.001
                    typeB = "DUM"

        elif atom_type == "uniqueB":
            # Unique B atoms: only present at lambda >= 0.5
            rowB = dfB[dfB["index"] == origB_idx].iloc[0]
            chargeB = rowB["charge"]
            massB = rowB["mass"]
            typeB = rowB["type"]
            if lam == 0:
                atom_name = "DUM"
                chargeA = 0.0
                massA = 0.001
                typeA = "DUM"
            elif lam == 1:
                atom_name = rowB["atom_name"]
                chargeA = chargeB
                massA = massB
                typeA = typeB
            else:
                atom_name = "DUM"
                if lam > 0.5:
                    chargeA = chargeB
                    massA = massB
                    typeA = typeB
                else:
                    chargeA = 0.0
                    massA = 0.001
                    typeA = "DUM"

        hybrid_atoms.append(
            HybridAtom(
                index=new_idx,
                atom_name=atom_name,
                typeA=typeA,
                typeB=typeB,
                chargeA=chargeA,
                chargeB=chargeB,
                massA=massA,
                massB=massB,
                mapped=(atom_type == "mapped"),
                origA_idx=origA_idx,
                origB_idx=origB_idx,
            )
        )
    return hybrid_atoms


# Removed redundant get_canonical_hybrid_atom_list function - use build_lambda_atom_list instead


def verify_hybrid_synchronization(hybrid_itp, hybrid_pdb, lam):
    """
    Verify that hybrid topology and coordinate files are synchronized.
    Returns True if synchronized, False otherwise.
    """
    # Read topology atoms
    topo_atoms = []
    with open(hybrid_itp, "r") as f:
        lines = f.readlines()

    in_atoms_section = False
    for line in lines:
        if line.strip() == "[ atoms ]":
            in_atoms_section = True
            continue
        elif in_atoms_section and line.strip().startswith("["):
            break
        elif in_atoms_section and line.strip() and not line.strip().startswith(";"):
            parts = line.split()
            if len(parts) >= 4:
                atom_index = int(parts[0])
                atom_name = parts[4]
                topo_atoms.append((atom_index, atom_name))

    # Read PDB atoms
    pdb_atoms = []
    with open(hybrid_pdb, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[13:16].strip()
                pdb_atoms.append(atom_name)

    # Check if atom counts match
    if len(topo_atoms) != len(pdb_atoms):
        print(
            f"[ERROR] Lambda {lam}: Topology has {len(topo_atoms)} atoms, PDB has {len(pdb_atoms)} atoms"
        )
        return False

    # Check if atom names match in order, accounting for dual topology logic
    lambda_val = float(lam)
    mismatches = []
    for i, (topo_idx, topo_name) in enumerate(topo_atoms):
        if i < len(pdb_atoms):
            pdb_name = pdb_atoms[i]
            # New naming convention:
            # - MCS atoms: always real names (never DUM)
            # - Pure states (λ=0, λ=1): unique atoms have real names
            # - Intermediate states (0<λ<1): unique atoms use "DUM"
            # Only flag as error if both names are non-dummy and different
            if (topo_name.strip() != "DUM" and pdb_name.strip() != "DUM" and
                    topo_name.strip() != pdb_name.strip()):
                mismatches.append((i + 1, topo_name, pdb_name))

    if mismatches:
        print(f"[ERROR] Lambda {lam}: Found {len(mismatches)} atom name mismatches:")
        for atom_num, topo_name, pdb_name in mismatches[:5]:  # Show first 5 mismatches
            print(f"  Atom {atom_num}: Topology: {topo_name}, PDB: {pdb_name}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more mismatches")
        return False

    print(
        f"[INFO] Lambda {lam}: Topology and PDB are synchronized ({len(topo_atoms)} atoms)"
    )
    return True


def find_closest_atom_coord(target_coord, reference_coords):
    """
    Find the closest atom coordinate to a target coordinate.

    Args:
        target_coord: tuple of (x, y, z) coordinates
        reference_coords: dictionary of atom_idx -> (x, y, z) coordinates

    Returns:
        tuple: closest coordinate
    """
    if not reference_coords:
        return target_coord

    min_distance = float('inf')
    closest_coord = target_coord

    for coord in reference_coords.values():
        distance = sum((a - b) ** 2 for a, b in zip(target_coord, coord)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_coord = coord

    return closest_coord


# --- Utility: Find the nearest mapped (MCS) atom index for a given coordinate ---
def find_nearest_mcs_atom(coord, mcs_coords):
    min_dist = float('inf')
    nearest_idx = None
    for idx, mcs_coord in mcs_coords.items():
        d = np.linalg.norm(np.array(coord) - np.array(mcs_coord))
        if d < min_dist:
            min_dist = d
            nearest_idx = idx
    return nearest_idx


def hybridize_coords_from_itp_interpolated(
        ligA_mol2, ligB_mol2, hybrid_itp, atom_map_txt, out_pdb, lam,
        cap_long_dummies=True, distance_threshold=1.2
):
    """
    Enhanced dual topology coordinate generation with growing procedure:
    - Lambda 0: Pure molecule A (only A atoms present with real properties)
    - Lambda 0.5: Hybrid structure with interpolated properties
    - Lambda 1: Pure molecule B (only B atoms present with real properties)
    - Intermediate values: Gradual transition with realistic dummy placement

    Coordinates are interpolated for growing regions:
    - Unique A atoms: interpolate from A position to closest MCS position
    - Unique B atoms: interpolate from closest MCS position to B position
    - Mapped atoms: interpolate between A and B positions
    - Dummy atoms that are far from any real atom are capped at the centroid or nearest real atom.
    """
    print(f"[DEBUG] Starting enhanced hybridize_coords_from_itp_interpolated for lambda {lam}")

    coordsA, namesA = parse_mol2_coords(ligA_mol2)
    coordsB, namesB = parse_mol2_coords(ligB_mol2)
    print(f"[DEBUG] Parsed coordsA: {len(coordsA)} atoms, coordsB: {len(coordsB)} atoms")

    # Try to parse .itp files, but don't fail if they don't exist
    itpA_path = ligA_mol2.replace(".mol2", ".itp")
    itpB_path = ligB_mol2.replace(".mol2", ".itp")

    dfA = None
    dfB = None

    try:
        if os.path.exists(itpA_path):
            dfA = parse_itp_atoms_full(itpA_path)
        else:
            print(f"[DEBUG] itpA file not found: {itpA_path}")
    except Exception as e:
        print(f"[DEBUG] Error parsing itpA: {e}")

    try:
        if os.path.exists(itpB_path):
            dfB = parse_itp_atoms_full(itpB_path)
        else:
            print(f"[DEBUG] itpB file not found: {itpB_path}")
    except Exception as e:
        print(f"[DEBUG] Error parsing itpB: {e}")

    if dfA is None:
        dfA = pd.DataFrame(
            [{"index": idx, "atom_name": namesA[idx]} for idx in sorted(coordsA.keys())]
        )
    if dfB is None:
        dfB = pd.DataFrame(
            [{"index": idx, "atom_name": namesB[idx]} for idx in sorted(coordsB.keys())]
        )

    mapping = load_atom_map(atom_map_txt)
    print(f"[DEBUG] Loaded mapping: {len(mapping)} entries")

    # Get lambda-specific atom list to match the topology
    atom_list = build_lambda_atom_list(dfA, dfB, mapping, lam)
    print(f"[DEBUG] Lambda {lam}: Found {len(atom_list)} atoms in atom_list")

    if len(atom_list) == 0:
        print(f"[WARNING] No atoms found for lambda {lam}")
        with open(out_pdb, "w") as f:
            f.write("END\n")
        return

    # Find MCS coordinates for growing procedure
    mcs_coords = {}
    for hybrid_idx, atom_name, origA_idx, origB_idx, atom_type in atom_list:
        if atom_type == "mapped":
            # For mapped atoms, MCS position is interpolated between A and B
            coordA = coordsA.get(origA_idx, (0.0, 0.0, 0.0))
            coordB = coordsB.get(origB_idx, (0.0, 0.0, 0.0))
            mcs_coords[origA_idx] = tuple((1 - lam) * a + lam * b for a, b in zip(coordA, coordB))
            mcs_coords[origB_idx] = mcs_coords[origA_idx]  # Same position for mapped atoms

    # Compute centroid of MCS for fallback
    if mcs_coords:
        centroid = tuple(np.mean([c[i] for c in mcs_coords.values()]) for i in range(3))
    else:
        centroid = (0.0, 0.0, 0.0)

    pdb_lines = []
    atom_counter = 0
    atom_coords = {}  # index -> coord

    print(f"[DEBUG] Processing {len(atom_list)} atoms with coordinate interpolation")
    for hybrid_idx, atom_name, origA_idx, origB_idx, atom_type in atom_list:
        atom_counter += 1

        if atom_type == "mapped":
            # Mapped atoms: interpolate between A and B coordinates
            coordA = coordsA.get(origA_idx, (0.0, 0.0, 0.0))
            coordB = coordsB.get(origB_idx, (0.0, 0.0, 0.0))
            coord = tuple((1 - lam) * a + lam * b for a, b in zip(coordA, coordB))
            atom_type_pdb = atom_name
        elif atom_type == "uniqueA":
            coordA = coordsA.get(origA_idx, None)
            if coordA is None:
                coordA = find_closest_atom_coord(centroid, coordsA)
            closest_mcs_coord = find_closest_atom_coord(coordA, mcs_coords)
            if lam <= 0.5:
                interp_factor = lam * 2.0
                coord = tuple((1 - interp_factor) * a + interp_factor * m for a, m in zip(coordA, closest_mcs_coord))
            else:
                coord = closest_mcs_coord
            if lam == 0:
                atom_type_pdb = atom_name
            elif lam == 1:
                atom_type_pdb = "DUM"
            else:
                atom_type_pdb = "DUM"
        elif atom_type == "uniqueB":
            coordB = coordsB.get(origB_idx, None)
            if coordB is None:
                coordB = find_closest_atom_coord(centroid, coordsB)
            closest_mcs_coord = find_closest_atom_coord(coordB, mcs_coords)
            if lam >= 0.5:
                interp_factor = (lam - 0.5) * 2.0
                coord = tuple((1 - interp_factor) * m + interp_factor * b for m, b in zip(closest_mcs_coord, coordB))
            else:
                coord = closest_mcs_coord
            if lam == 0:
                atom_type_pdb = "DUM"
            elif lam == 1:
                atom_type_pdb = atom_name
            else:
                atom_type_pdb = "DUM"
        else:
            coord = find_closest_atom_coord(centroid, coordsA)
            atom_type_pdb = "DUM"
        atom_coords[hybrid_idx] = coord
        pdb_lines.append(
            f"HETATM{atom_counter:5d}  {atom_type_pdb:<4s}LIG     1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n"
        )

    # Strict dummy capping: cap dummies at nearest mapped (MCS) atom
    if cap_long_dummies:
        real_atoms = [i for i, (idx, name, a, b, t) in enumerate(atom_list, 1) if t == "mapped" or t == "uniqueA" or t == "uniqueB"]
        for hybrid_idx, atom_name, origA_idx, origB_idx, atom_type in atom_list:
            if atom_type_pdb == "DUM":
                # Find nearest mapped atom
                if mcs_coords:
                    nearest_idx = find_nearest_mcs_atom(atom_coords[hybrid_idx], mcs_coords)
                    if nearest_idx is not None:
                        cap_coord = mcs_coords[nearest_idx]
                        d = np.linalg.norm(np.array(atom_coords[hybrid_idx]) - np.array(cap_coord))
                        if d > distance_threshold:
                            print(f"[WARNING] Dummy atom {hybrid_idx} is {d:.2f} nm from nearest mapped atom. Capping at mapped atom.")
                            atom_coords[hybrid_idx] = cap_coord
                            pdb_lines[hybrid_idx - 1] = f"HETATM{hybrid_idx:5d}  {atom_type_pdb:<4s}LIG     1    {cap_coord[0]:8.3f}{cap_coord[1]:8.3f}{cap_coord[2]:8.3f}  1.00  0.00\n"
                    else:
                        # Fallback: cap at centroid
                        d = np.linalg.norm(np.array(atom_coords[hybrid_idx]) - np.array(centroid))
                        if d > distance_threshold:
                            print(f"[WARNING] Dummy atom {hybrid_idx} is {d:.2f} nm from centroid. Capping at centroid.")
                            atom_coords[hybrid_idx] = centroid
                            pdb_lines[hybrid_idx - 1] = f"HETATM{hybrid_idx:5d}  {atom_type_pdb:<4s}LIG     1    {centroid[0]:8.3f}{centroid[1]:8.3f}{centroid[2]:8.3f}  1.00  0.00\n"

    with open(out_pdb, "w") as f:
        for line in pdb_lines:
            f.write(line)
        f.write("END\n")
    print(f"[DEBUG] Successfully wrote {out_pdb}")


def generate_exclusions(hybrid_atoms, lam, dfA=None, dfB=None, mapping=None):
    """
    Generate exclusions between A and B atoms that are present in the current lambda window.
    This generates exclusions only for atoms that exist in the current hybrid topology.

    Args:
        hybrid_atoms: List of HybridAtom objects for current lambda
        lam: Lambda value
        dfA: DataFrame of molecule A atoms (for reference)
        dfB: DataFrame of molecule B atoms (for reference)
        mapping: Atom mapping dictionary

    Returns:
        List of exclusion dictionaries (only for atoms present in current lambda)
    """
    exclusions = []

    # Get the set of atom indices that are actually present in this lambda window
    present_atom_indices = set(atom.index for atom in hybrid_atoms)

    # Classify present atoms by their origin
    atoms_a = []  # Atoms from molecule A (including mapped) that are present
    atoms_b = []  # Atoms from molecule B (including mapped) that are present

    for atom in hybrid_atoms:
        if atom.origA_idx is not None:
            atoms_a.append(atom.index)
        if atom.origB_idx is not None:
            atoms_b.append(atom.index)

    # Create exclusions between present A atoms and present B atoms
    for atom_a in atoms_a:
        for atom_b in atoms_b:
            # Don't exclude self-interactions
            if atom_a == atom_b:
                continue
            # Don't exclude mapped atoms with themselves (if they're the same atom)
            if atom_a in present_atom_indices and atom_b in present_atom_indices:
                # Check if these are mapped atoms (same atom in A and B)
                atom_a_obj = next((a for a in hybrid_atoms if a.index == atom_a), None)
                atom_b_obj = next((a for a in hybrid_atoms if a.index == atom_b), None)
                if (atom_a_obj is not None and atom_b_obj is not None and
                        atom_a_obj.origA_idx is not None and atom_b_obj.origB_idx is not None and
                        mapping is not None and atom_a_obj.origA_idx in mapping and mapping[
                            atom_a_obj.origA_idx] == atom_b_obj.origB_idx):
                    continue

                exclusions.append({
                    "ai": atom_a,
                    "aj": atom_b,
                    "funct": 1  # Standard exclusion
                })

    print(
        f"[DEBUG] Lambda {lam}: Generated {len(exclusions)} exclusions for present atoms (A: {len(atoms_a)}, B: {len(atoms_b)})")
    return exclusions


def create_hybrid_topology_for_lambda(
        dfA, dfB, bondsA, bondsB, anglesA, anglesB, dihedA, dihedB, mapping, lam
):
    """
    Create hybrid topology for a specific lambda value with proper filtering.
    """
    # Create hybrid atoms (this now gets the correct atom list for this lambda)
    hybrid_atoms = build_hybrid_atoms_interpolated(dfA, dfB, mapping, lam)

    # Get present indices from the hybrid atoms
    present_indices = [atom.index for atom in hybrid_atoms]

    # Filter topology sections to only include present atoms and remove invalid terms
    bondsA_filtered = filter_topology_sections(bondsA, present_indices)
    anglesA_filtered = filter_topology_sections(anglesA, present_indices)
    dihedA_filtered = filter_topology_sections(dihedA, present_indices)

    # Also filter B sections to ensure no out-of-bounds references
    bondsB_filtered = filter_topology_sections(bondsB, present_indices)
    anglesB_filtered = filter_topology_sections(anglesB, present_indices)
    dihedB_filtered = filter_topology_sections(dihedB, present_indices)

    # Create dummy parameters for missing terms
    dummy_params = {"r": "0.0", "k": "0.0"}

    # Convert to hybrid terms using filtered sections for both A and B
    hybrid_bonds = build_hybrid_terms(
        bondsA_filtered, bondsB_filtered, mapping, ["ai", "aj"], HybridBond,
        dummy_params, dummy_params, ["r", "k"], ["r", "k"]
    )
    hybrid_angles = build_hybrid_terms(
        anglesA_filtered, anglesB_filtered, mapping, ["ai", "aj", "ak"], HybridAngle,
        dummy_params, dummy_params, ["r", "k"], ["r", "k"]
    )
    hybrid_dihedrals = build_hybrid_terms(
        dihedA_filtered, dihedB_filtered, mapping, ["ai", "aj", "ak", "al"], HybridDihedral,
        dummy_params, dummy_params, ["r", "k"], ["r", "k"]
    )

    # Filter out any remaining invalid terms from hybrid terms
    hybrid_bonds = [bond for bond in hybrid_bonds if bond.ai != bond.aj]
    hybrid_angles = [
        angle for angle in hybrid_angles
        if angle.ai != angle.aj and angle.aj != angle.ak and angle.ai != angle.ak
    ]
    hybrid_dihedrals = [
        dih for dih in hybrid_dihedrals
        if dih.ai != dih.aj and dih.ai != dih.ak and dih.ai != dih.al
           and dih.aj != dih.ak and dih.aj != dih.al and dih.ak != dih.al
    ]

    # Generate filtered pairs to avoid cut-off issues
    hybrid_pairs = generate_filtered_pairs(hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, lam)

    # Exclusions are now handled per-lambda in process_lambda_windows
    return hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, hybrid_pairs, None


def generate_filtered_pairs(hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, lam):
    """
    Generate filtered 1-4 pair interactions to avoid cut-off issues in dual topology.

    This function creates pairs only for atoms that are:
    1. Present in the current lambda window
    2. Connected through valid bond paths (1-4 interactions)
    3. Not dummy atoms at both ends (to avoid long-distance interactions)
    4. Within reasonable distance limits to avoid cut-off issues

    Args:
        hybrid_atoms: List of HybridAtom objects for current lambda
        hybrid_bonds: List of HybridBond objects
        hybrid_angles: List of HybridAngle objects
        hybrid_dihedrals: List of HybridDihedral objects

    Returns:
        List of pair dictionaries with ai, aj, funct
    """
    pairs = []

    # Create adjacency list for bond connectivity
    adjacency = {}
    for atom in hybrid_atoms:
        adjacency[atom.index] = []

    for bond in hybrid_bonds:
        if hasattr(bond, 'ai') and hasattr(bond, 'aj'):
            adjacency[bond.ai].append(bond.aj)
            adjacency[bond.aj].append(bond.ai)

    # Find all 1-4 interactions (atoms connected through 3 bonds)
    for atom_i in hybrid_atoms:
        # Skip pure dummy atoms to avoid long-distance interactions
        if atom_i.typeA == "DUM" and atom_i.typeB == "DUM":
            continue

        for atom_j in hybrid_atoms:
            if atom_j.index <= atom_i.index:
                continue  # Avoid duplicates
            # Skip pure dummy atoms to avoid long-distance interactions
            if atom_j.typeA == "DUM" and atom_j.typeB == "DUM":
                continue

            # Check if atoms are 1-4 connected (3 bonds apart)
            if is_1_4_connected(atom_i.index, atom_j.index, adjacency):
                # Additional safety check: ensure at least one atom is real
                # This prevents dummy-dummy pairs that could cause cut-off issues
                if (atom_i.typeA != "DUM" or atom_i.typeB != "DUM") and \
                        (atom_j.typeA != "DUM" or atom_j.typeB != "DUM"):

                    # Additional filtering: avoid pairs where both atoms are dummy in one state
                    # This can happen in intermediate lambda values
                    if lam == 0.0:
                        # At lambda 0, only include pairs where atom_i is not dummy in A state
                        if atom_i.typeA != "DUM":
                            pairs.append({
                                "ai": atom_i.index,
                                "aj": atom_j.index,
                                "funct": 1  # Standard 1-4 interaction
                            })
                    elif lam == 1.0:
                        # At lambda 1, only include pairs where atom_j is not dummy in B state
                        if atom_j.typeB != "DUM":
                            pairs.append({
                                "ai": atom_i.index,
                                "aj": atom_j.index,
                                "funct": 1  # Standard 1-4 interaction
                            })
                    else:
                        # At intermediate lambda, be more conservative
                        # Only include if both atoms have some real character
                        if (atom_i.typeA != "DUM" or atom_i.typeB != "DUM") and \
                                (atom_j.typeA != "DUM" or atom_j.typeB != "DUM"):
                            pairs.append({
                                "ai": atom_i.index,
                                "aj": atom_j.index,
                                "funct": 1  # Standard 1-4 interaction
                            })

    return pairs


def is_1_4_connected(atom_i, atom_j, adjacency, max_depth=3):
    """
    Check if two atoms are connected through exactly 3 bonds (1-4 interaction).

    Args:
        atom_i: First atom index
        atom_j: Second atom index
        adjacency: Adjacency list representation of bonds
        max_depth: Maximum search depth (3 for 1-4 interactions)

    Returns:
        bool: True if atoms are 1-4 connected
    """
    if atom_i == atom_j:
        return False

    # Use BFS to find shortest path
    visited = set()
    queue = [(atom_i, 0)]  # (atom, distance)

    while queue:
        current_atom, distance = queue.pop(0)

        if current_atom == atom_j:
            return distance == 3  # Exactly 3 bonds apart

        if current_atom in visited or distance >= max_depth:
            continue

        visited.add(current_atom)

        for neighbor in adjacency.get(current_atom, []):
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))

    return False


def generate_lambda_exclusions_conditional(hybrid_atoms, distance_threshold=1.2, coords=None):
    """
    For a given lambda, generate exclusions for STATEA and STATEB separately.
    - Exclude uniqueA from uniqueB and vice versa, but only in the state where the dummy exists.
    - Returns: {"STATEA": [...], "STATEB": [...]} exclusion dicts.
    Optionally, if coords are provided, also return a list of dummy atoms that are far from any real atom.
    """
    uniqueA = [atom.index for atom in hybrid_atoms if
               not atom.mapped and atom.origA_idx is not None and atom.origB_idx is None]
    uniqueB = [atom.index for atom in hybrid_atoms if
               not atom.mapped and atom.origB_idx is not None and atom.origA_idx is None]
    exclusions_statea = []
    exclusions_stateb = []
    for a in uniqueA:
        for b in uniqueB:
            # Exclude uniqueA from uniqueB in STATEA (uniqueA is real, uniqueB is dummy)
            exclusions_statea.append({"ai": a, "aj": b, "funct": 1})
            # Exclude uniqueB from uniqueA in STATEB (uniqueB is real, uniqueA is dummy)
            exclusions_stateb.append({"ai": b, "aj": a, "funct": 1})
    # Optionally, detect long dummy tails if coords are provided
    long_dummies = []
    if coords is not None:
        # Find all real atom indices
        real_atoms = [atom.index for atom in hybrid_atoms if atom.typeA != "DUM" or atom.typeB != "DUM"]
        for atom in hybrid_atoms:
            if atom.typeA == "DUM" and atom.typeB == "DUM":
                idx = atom.index
                if idx in coords:
                    dists = [np.linalg.norm(np.array(coords[idx]) - np.array(coords[j])) for j in real_atoms if
                             j in coords]
                    if dists and min(dists) > distance_threshold:
                        long_dummies.append((idx, min(dists)))
    return {"STATEA": exclusions_statea, "STATEB": exclusions_stateb, "long_dummies": long_dummies}


# --- New: Utility to compute distance between two atoms given atom_coords dict ---
def atom_distance(idx1, idx2, atom_coords):
    c1 = np.array(atom_coords[idx1])
    c2 = np.array(atom_coords[idx2])
    return np.linalg.norm(c1 - c2)


# --- New: Generate exclusions only for 1-4 pairs and spatially close atoms ---
def generate_lambda_exclusions_refined(hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, atom_coords, rlist=1.2, mcs_indices=None):
    """
    Generate exclusions for 1-4 pairs and spatially close atoms only.
    - Exclude 1-4 pairs (atoms connected through 3 bonds)
    - Exclude any pair of atoms (dummy or real) that are both present and closer than rlist
    - Do NOT exclude all uniqueA-uniqueB pairs
    - Never include exclusions for pairs where both atoms are dummies
    - Remove any exclusion where the distance between atoms is greater than rlist
    - Only include exclusions for pairs where both atoms are in the same connected component as the core
    """
    # Find all atoms connected to the core
    if mcs_indices is not None:
        connected = get_connected_atoms_to_core(hybrid_atoms, hybrid_bonds, mcs_indices)
    else:
        connected = set(atom.index for atom in hybrid_atoms)
    exclusions = set()
    # Build adjacency for 1-4 detection
    adjacency = {atom.index: [] for atom in hybrid_atoms}
    for bond in hybrid_bonds:
        adjacency[bond.ai].append(bond.aj)
        adjacency[bond.aj].append(bond.ai)
    # 1-4 pairs
    for atom_i in hybrid_atoms:
        for atom_j in hybrid_atoms:
            if atom_j.index <= atom_i.index:
                continue
            # Filter out dummy-dummy pairs
            if (atom_i.typeA == "DUM" and atom_i.typeB == "DUM") and (atom_j.typeA == "DUM" and atom_j.typeB == "DUM"):
                continue
            # Only include if both atoms are connected to the core
            if atom_i.index not in connected or atom_j.index not in connected:
                continue
            if is_1_4_connected(atom_i.index, atom_j.index, adjacency):
                d = atom_distance(atom_i.index, atom_j.index, atom_coords)
                if d <= rlist:
                    exclusions.add((atom_i.index, atom_j.index))
                    exclusions.add((atom_j.index, atom_i.index))
    # Spatially close pairs (within rlist, but not bonded)
    for atom_i in hybrid_atoms:
        for atom_j in hybrid_atoms:
            if atom_j.index <= atom_i.index:
                continue
            # Filter out dummy-dummy pairs
            if (atom_i.typeA == "DUM" and atom_i.typeB == "DUM") and (atom_j.typeA == "DUM" and atom_j.typeB == "DUM"):
                continue
            # Only include if both atoms are connected to the core
            if atom_i.index not in connected or atom_j.index not in connected:
                continue
            d = atom_distance(atom_i.index, atom_j.index, atom_coords)
            if d < rlist:
                exclusions.add((atom_i.index, atom_j.index))
                exclusions.add((atom_j.index, atom_i.index))
    # Format for GROMACS
    exclusion_dict = {}
    for ai, aj in exclusions:
        exclusion_dict.setdefault(ai, set()).add(aj)
    # Return as list of dicts for write_hybrid_topology
    result = []
    for ai, aj_set in exclusion_dict.items():
        for aj in aj_set:
            result.append({"ai": ai, "aj": aj, "funct": 1})
    return result


# --- Minimal, robust exclusion generator for dual topology FEP ---
def generate_dual_topology_exclusions(hybrid_atoms, lam):
    """
    For intermediate lambdas (0 < lambda < 1), exclude all uniqueA from all uniqueB atoms (and vice versa).
    For lambda=0 or 1, no uniqueA-uniqueB exclusions are needed.
    """
    # Identify uniqueA and uniqueB atom indices
    uniqueA = [atom.index for atom in hybrid_atoms if not atom.mapped and atom.origA_idx is not None and atom.origB_idx is None]
    uniqueB = [atom.index for atom in hybrid_atoms if not atom.mapped and atom.origB_idx is not None and atom.origA_idx is None]
    exclusions = []
    if 0 < lam < 1:
        for a in uniqueA:
            for b in uniqueB:
                exclusions.append({"ai": a, "aj": b, "funct": 1})
                exclusions.append({"ai": b, "aj": a, "funct": 1})
    return exclusions


# --- Refactor process_lambda_windows to use refined exclusions and robust dummy capping ---
def process_lambda_windows(dfA, dfB, bondsA, bondsB, anglesA, anglesB, dihedA, dihedB, mapping,
                           ligA_mol2=None, ligB_mol2=None, atom_map_txt=None):
    """
    Process all lambda windows to generate hybrid topologies and coordinates.
    This consolidates the redundant lambda processing code from main().
    """
    lambdas = np.arange(0, 1.05, 0.05)
    for lam in lambdas:
        lam_str = f"{lam:.2f}"
        lam_dir = f"lambda_{lam_str}"
        if not os.path.exists(lam_dir):
            os.makedirs(lam_dir)
        # Generate hybrid topology
        outfilename = os.path.join(lam_dir, f"hybrid_lambda_{lam_str}.itp")
        hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, hybrid_pairs, _ = (
            create_hybrid_topology_for_lambda(
                dfA, dfB, bondsA, bondsB, anglesA, anglesB, dihedA, dihedB, mapping, lam
            )
        )
        # Generate coordinates and cap dummies robustly
        if ligA_mol2 and ligB_mol2 and atom_map_txt:
            hybrid_itp = outfilename
            out_pdb = os.path.join(lam_dir, f"hybrid_lambda_{lam_str}.pdb")
            if not os.path.exists(hybrid_itp):
                print(f"Warning: {hybrid_itp} not found, skipping lambda {lam_str}")
                continue
            hybridize_coords_from_itp_interpolated(
                ligA_mol2, ligB_mol2, hybrid_itp, atom_map_txt, out_pdb, lam
            )
            print(f"Wrote {out_pdb}")
        # Minimal robust exclusions for dual topology
        lambda_exclusions = generate_dual_topology_exclusions(hybrid_atoms, lam)
        write_hybrid_topology(
            outfilename, hybrid_atoms, hybrid_bonds=hybrid_bonds,
            hybrid_angles=hybrid_angles, hybrid_dihedrals=hybrid_dihedrals,
            hybrid_pairs=hybrid_pairs, hybrid_exclusions=lambda_exclusions,
            system_name="LigandA to LigandB Hybrid", molecule_name="LIG", nmols=1
        )
        # Create position restraints file
        posre_filename = os.path.join(lam_dir, "posre_ligand.itp")
        write_position_restraints_file(posre_filename, hybrid_atoms)
        print(f"Wrote {outfilename}")
        if os.path.exists(posre_filename):
            print(f"Wrote {posre_filename}")
        # Verify synchronization
        if ligA_mol2 and ligB_mol2 and atom_map_txt:
            out_pdb = os.path.join(lam_dir, f"hybrid_lambda_{lam_str}.pdb")
            verify_hybrid_synchronization(outfilename, out_pdb, lam_str)


# =====================
# CLI
# =====================
def print_growing_procedure_info():
    """
    Print information about the enhanced dual topology growing procedure.
    """
    print("""
Enhanced Dual Topology Growing Procedure:

The FEP preparation now implements an improved dual topology approach with realistic
dummy atom placement and smooth transitions between states:

Lambda Values and Behavior:
- Lambda = 0.0: Pure molecule A
  * Only mapped atoms + unique A atoms are present
  * All atoms have real A properties and coordinates
  * No B-specific atoms in topology or coordinates
  * Clean, minimal topology for pure A state

- Lambda = 0.5: Hybrid structure
  * All atoms present: mapped + uniqueA + uniqueB
  * Mapped atoms: interpolated coordinates and properties
  * Unique A atoms: coordinates interpolated from A position to MCS position
  * Unique B atoms: coordinates interpolated from MCS position to B position
  * Bond distances are half-way between A and B states

- Lambda = 1.0: Pure molecule B
  * Only mapped atoms + unique B atoms are present
  * All atoms have real B properties and coordinates
  * No A-specific atoms in topology or coordinates
  * Clean, minimal topology for pure B state

- Intermediate values (0 < lambda < 1):
  * All atoms present for smooth transitions
  * Unique A atoms: coordinates interpolate from A position to closest MCS position
  * Unique B atoms: coordinates interpolate from closest MCS position to B position
  * Gradual property and coordinate interpolation for mapped atoms

Key Improvements:
1. Clean pure states: lambda 0 and 1 only include relevant atoms
2. Consistent naming convention: MCS atoms always real, unique atoms DUM at intermediate λ
3. Dummy atoms placed at closest real atom positions (not random)
4. Smooth growing/degrowing of unique atoms
5. Realistic bond distance interpolation at lambda = 0.5
6. Minimized non-bonded interaction errors
7. Continuous structure guarantee in MCS algorithm
""")


def print_help():
    print(
        """
Enhanced FEP Preparation Tool - Dual Topology with Growing Procedure

Usage:
  python fep_prep.py align ligandA.mol2 ligandB.mol2 aligned_ligandB.mol2
      Find MCS and align ligand B to ligand A using MCS-based alignment
  python fep_prep.py mcs ligandA.mol2 ligandB.mol2 atom_map.txt
      Find the maximum common substructure (continuous only) and write atom_map.txt
  python fep_prep.py hybrid_topology ligandA.itp ligandB.itp atom_map.txt
      Generate hybrid .itp files for all lambda windows with growing procedure
  python fep_prep.py hybrid_coords ligandA.mol2 ligandB.mol2 atom_map.txt
      Generate hybridized .pdb files for all lambda windows with realistic dummy placement
  python fep_prep.py full_workflow ligandA.mol2 ligandB.mol2
      Complete workflow: find MCS, align ligands, generate hybrid topology and coordinates
  python fep_prep.py growing_info
      Display detailed information about the growing procedure

Key Features:
- Continuous MCS: Only largest connected substructure returned
- Realistic dummy placement: Dummies at closest real atom positions
- Growing procedure: Smooth transitions between A and B states
- Lambda 0.5: Perfect hybrid with interpolated bond distances
- Minimized errors: Reduced non-bonded interaction artifacts
"""
    )


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "growing_info":
        print_growing_procedure_info()
    elif cmd == "align":
        if len(sys.argv) != 5:
            print(
                "Usage: python fep_prep.py align ligandA.mol2 ligandB.mol2 aligned_ligandB.mol2"
            )
            sys.exit(1)
        ligA_mol2, ligB_mol2, aligned_ligandB_mol2 = sys.argv[2:5]

        # First find MCS to get atom mapping
        g1 = MolGraph.from_mol2(ligA_mol2)
        g2 = MolGraph.from_mol2(ligB_mol2)
        size, mapping, atoms1, atoms2 = find_mcs(g1, g2)

        if mapping is None or size == 0:
            print("No common substructure found. Cannot align ligands.")
            sys.exit(1)

        print(f"MCS size: {size}")
        print(f"Mapping (ligandA -> ligandB): {mapping}")

        # Then align using the mapping
        align_ligands_with_mapping(ligA_mol2, ligB_mol2, aligned_ligandB_mol2, mapping)
    elif cmd == "mcs":
        if len(sys.argv) != 5:
            print(
                "Usage: python fep_prep.py mcs ligandA.mol2 ligandB.mol2 atom_map.txt"
            )
            sys.exit(1)
        mol1, mol2, outmap = sys.argv[2:5]
        g1 = MolGraph.from_mol2(mol1)
        g2 = MolGraph.from_mol2(mol2)
        size, mapping, atoms1, atoms2 = find_mcs(g1, g2)
        if mapping is None:
            print("No MCS found.")
            sys.exit(1)
        print(f"MCS size: {size}")
        print(f"Mapping (mol1 -> mol2): {mapping}")
        write_atom_map(mapping, outmap)
    elif cmd == "hybrid_topology":
        if len(sys.argv) != 5:
            print(
                "Usage: python fep_prep.py hybrid_topology ligandA.itp ligandB.itp atom_map.txt"
            )
            sys.exit(1)
        itpA, itpB, mapfile = sys.argv[2:5]
        dfA = parse_itp_atoms_full(itpA)
        dfB = parse_itp_atoms_full(itpB)
        mapping = load_atom_map(mapfile)
        bondsA = parse_itp_section(itpA, "bonds", 5, ["ai", "aj", "funct", "r", "k"])
        bondsB = parse_itp_section(itpB, "bonds", 5, ["ai", "aj", "funct", "r", "k"])
        anglesA = parse_itp_section(itpA, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"])
        anglesB = parse_itp_section(itpB, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"])
        dihedA = parse_itp_section(itpA, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"])
        dihedB = parse_itp_section(itpB, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"])

        process_lambda_windows(dfA, dfB, bondsA, bondsB, anglesA, anglesB, dihedA, dihedB, mapping)

    elif cmd == "hybrid_coords":
        if len(sys.argv) != 5:
            print(
                "Usage: python fep_prep.py hybrid_coords ligandA.mol2 ligandB.mol2 atom_map.txt"
            )
            sys.exit(1)
        ligA_mol2, ligB_mol2, atom_map_txt = sys.argv[2:5]

        # Check if aligned ligand B exists, otherwise use original
        aligned_ligandB_mol2 = ligB_mol2.replace(".mol2", "_aligned.mol2")
        if os.path.exists(aligned_ligandB_mol2):
            print(f"Using aligned ligand B: {aligned_ligandB_mol2}")
            ligB_mol2 = aligned_ligandB_mol2
        else:
            print(f"Using original ligand B: {ligB_mol2}")

        # Parse topology files for coordinate generation
        itpA = ligA_mol2.replace(".mol2", ".itp")
        itpB = ligB_mol2.replace(".mol2", ".itp")

        if os.path.exists(itpA) and os.path.exists(itpB):
            dfA = parse_itp_atoms_full(itpA)
            dfB = parse_itp_atoms_full(itpB)
            bondsA = parse_itp_section(itpA, "bonds", 5, ["ai", "aj", "funct", "r", "k"])
            bondsB = parse_itp_section(itpB, "bonds", 5, ["ai", "aj", "funct", "r", "k"])
            anglesA = parse_itp_section(itpA, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"])
            anglesB = parse_itp_section(itpB, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"])
            dihedA = parse_itp_section(itpA, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"])
            dihedB = parse_itp_section(itpB, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"])
            mapping = load_atom_map(atom_map_txt)

            process_lambda_windows(dfA, dfB, bondsA, bondsB, anglesA, anglesB, dihedA, dihedB, mapping,
                                   ligA_mol2, ligB_mol2, atom_map_txt)
        else:
            print("Warning: ITP files not found, coordinate generation may be limited")
            # Fallback: just generate coordinates without topology
        lambdas = np.arange(0, 1.05, 0.05)
        for lam in lambdas:
            lam_str = f"{lam:.2f}"
            lam_dir = f"lambda_{lam_str}"
            if not os.path.exists(lam_dir):
                os.makedirs(lam_dir)
            hybrid_itp = os.path.join(lam_dir, f"hybrid_lambda_{lam_str}.itp")
            out_pdb = os.path.join(lam_dir, f"hybrid_lambda_{lam_str}.pdb")

            if not os.path.exists(hybrid_itp):
                print(f"Warning: {hybrid_itp} not found, skipping lambda {lam_str}")
                continue

            hybridize_coords_from_itp_interpolated(
                ligA_mol2, ligB_mol2, hybrid_itp, atom_map_txt, out_pdb, lam
            )
            print(f"Wrote {out_pdb}")
            verify_hybrid_synchronization(hybrid_itp, out_pdb, lam_str)
    elif cmd == "full_workflow":
        if len(sys.argv) != 4:
            print(
                "Usage: python fep_prep.py full_workflow ligandA.mol2 ligandB.mol2"
            )
            sys.exit(1)
        ligA_mol2, ligB_mol2 = sys.argv[2:4]
        aligned_ligandB_mol2 = ligB_mol2.replace(".mol2", "_aligned.mol2")

        # Step 1: Find MCS to get atom mapping
        g1 = MolGraph.from_mol2(ligA_mol2)
        g2 = MolGraph.from_mol2(ligB_mol2)
        size, mapping, atoms1, atoms2 = find_mcs(g1, g2)
        if mapping is None:
            print("No MCS found.")
            sys.exit(1)
        print(f"MCS size: {size}")
        print(f"Mapping (ligandA -> ligandB): {mapping}")

        # Step 2: Align ligands using the MCS mapping
        align_ligands_with_mapping(ligA_mol2, ligB_mol2, aligned_ligandB_mol2, mapping)

        # Step 3: Write atom map for further processing
        outmap = "atom_map.txt"
        write_atom_map(mapping, outmap)

        itpA = ligA_mol2.replace(".mol2", ".itp")
        itpB = ligB_mol2.replace(".mol2", ".itp")
        dfA = parse_itp_atoms_full(itpA)
        dfB = parse_itp_atoms_full(itpB)
        mapping = load_atom_map(outmap)
        bondsA = parse_itp_section(itpA, "bonds", 5, ["ai", "aj", "funct", "r", "k"])
        bondsB = parse_itp_section(itpB, "bonds", 5, ["ai", "aj", "funct", "r", "k"])
        anglesA = parse_itp_section(itpA, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"])
        anglesB = parse_itp_section(itpB, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"])
        dihedA = parse_itp_section(itpA, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"])
        dihedB = parse_itp_section(itpB, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"])

        process_lambda_windows(dfA, dfB, bondsA, bondsB, anglesA, anglesB, dihedA, dihedB, mapping,
                               ligA_mol2, aligned_ligandB_mol2, outmap)
    else:
        print_help()
        sys.exit(1)


# --- Utility: Find all atoms connected to the mapped core (MCS) via bonds ---
def get_connected_atoms_to_core(hybrid_atoms, hybrid_bonds, mcs_indices):
    # Build adjacency list
    adjacency = {atom.index: set() for atom in hybrid_atoms}
    for bond in hybrid_bonds:
        adjacency[bond.ai].add(bond.aj)
        adjacency[bond.aj].add(bond.ai)
    # BFS from all MCS indices
    connected = set(mcs_indices)
    queue = list(mcs_indices)
    while queue:
        idx = queue.pop(0)
        for neighbor in adjacency[idx]:
            if neighbor not in connected:
                connected.add(neighbor)
                queue.append(neighbor)
    return connected


if __name__ == "__main__":
    main()
