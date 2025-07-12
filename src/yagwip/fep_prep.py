# === Standard Library Imports ===
import sys
from collections import defaultdict
import logging

# === Third-Party Imports ===
import pandas as pd
import numpy as np

logger = logging.getLogger("hybrid_topology")


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


def build_hybrid_atoms(dfA, dfB, mapping):
    hybrid_atoms = []
    all_indices = set(dfA["index"]).union(dfB["index"])
    for idx in sorted(all_indices):
        inA = idx in dfA["index"].values
        inB = idx in mapping and mapping[idx] in dfB["index"].values
        if inA and inB:
            rowA = dfA[dfA["index"] == idx].iloc[0]
            idxB = mapping[idx]
            rowB = dfB[dfB["index"] == idxB].iloc[0]
            hybrid_atoms.append(
                HybridAtom(
                    index=idx,
                    atom_name=rowA["atom_name"],
                    typeA=rowA["type"],
                    typeB=rowB["type"],
                    chargeA=rowA["charge"],
                    chargeB=rowB["charge"],
                    massA=rowA["mass"],
                    massB=rowB["mass"],
                    mapped=True,
                )
            )
        elif inA:
            rowA = dfA[dfA["index"] == idx].iloc[0]
            logger.warning(
                f"Dummy atom in state B: index {idx}, type {rowA['type']}, mass {rowA['mass']}"
            )
            hybrid_atoms.append(
                HybridAtom(
                    index=idx,
                    atom_name=rowA["atom_name"],
                    typeA=rowA["type"],
                    typeB="DUM",
                    chargeA=rowA["charge"],
                    chargeB=0.0,
                    massA=rowA["mass"],
                    massB=0.1,
                    mapped=False,
                )
            )
        else:
            idxB = None
            for k, v in mapping.items():
                if v == idx:
                    idxB = v
                    break
            if idxB is not None:
                rowB = dfB[dfB["index"] == idxB].iloc[0]
            else:
                rowB = dfB[dfB["index"] == idx].iloc[0]
            logger.warning(
                f"Dummy atom in state A: index {idx}, type {rowB['type']}, mass {rowB['mass']}"
            )
            hybrid_atoms.append(
                HybridAtom(
                    index=idx,
                    atom_name=rowB["atom_name"],
                    typeA="DUM",
                    typeB=rowB["type"],
                    chargeA=0.0,
                    chargeB=rowB["charge"],
                    massA=0.1,
                    massB=rowB["mass"],
                    mapped=False,
                )
            )
    # Validation: check for mapped atoms with zero mass
    for atom in hybrid_atoms:
        if atom.mapped and (atom.massA == 0.0 or atom.massB == 0.0):
            logger.error(
                f"Mapped atom {atom.index} has zero mass! massA={atom.massA}, massB={atom.massB}"
            )
    return hybrid_atoms


def write_hybrid_topology(
        filename,
        hybrid_atoms,
        hybrid_bonds=None,
        hybrid_pairs=None,
        hybrid_angles=None,
        hybrid_dihedrals=None,
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

        # Add conditional include for position restraints
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
    For each lambda, build a new hybrid atom list with correct dual topology logic:
    - Lambda 0: mapped + uniqueA atoms (A atoms are real, B atoms are dummies)
    - Lambda 1: mapped + uniqueB atoms (B atoms are real, A atoms are dummies)
    - 0 < lambda < 1: mapped + uniqueA + uniqueB atoms (all atoms present, some are dummies)
    - Mapped atoms: interpolate charge/mass/type.
    - Unique to A: typeB='DUM', chargeB=0, massB=0.1 (dummy mass zero).
    - Unique to B: typeA='DUM', chargeA=0, massA=0.1 (dummy mass zero).
    """
    # Get the lambda-specific atom list
    atom_list = build_lambda_atom_list(dfA, dfB, mapping, lam)
    hybrid_atoms = []
    for new_idx, (old_idx, atom_name, origA_idx, origB_idx, atom_type) in enumerate(
            atom_list, 1
    ):
        if atom_type == "mapped":
            rowA = dfA[dfA["index"] == origA_idx].iloc[0]
            rowB = dfB[dfB["index"] == origB_idx].iloc[0]
            chargeA = rowA["charge"]
            chargeB = rowB["charge"]
            massA = rowA["mass"]
            massB = rowB["mass"]
            typeA = rowA["type"]
            typeB = rowB["type"]
        elif atom_type == "uniqueA":
            rowA = dfA[dfA["index"] == origA_idx].iloc[0]
            chargeA = rowA["charge"]
            massA = rowA["mass"]
            typeA = rowA["type"]
            chargeB = 0.0
            massB = 0.1  # Dummy gets zero mass in decoupled state
            typeB = "DUM"
        elif atom_type == "uniqueB":
            rowB = dfB[dfB["index"] == origB_idx].iloc[0]
            chargeA = 0.0
            massA = 0.1  # Dummy gets zero mass in decoupled state
            typeA = "DUM"
            chargeB = rowB["charge"]
            massB = rowB["mass"]
            typeB = rowB["type"]
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


def get_canonical_hybrid_atom_list(dfA, dfB, mapping):
    """
    Returns a canonical hybrid atom list:
    - mapped_atoms: list of (hybrid_index, atom_name, origA_idx, origB_idx, 'mapped')
    - uniqueA_atoms: list of (hybrid_index, atom_name, origA_idx, None, 'uniqueA')
    - uniqueB_atoms: list of (hybrid_index, atom_name, None, origB_idx, 'uniqueB')
    Order: mapped (A order), uniqueA (A order), uniqueB (B order)
    """
    mapped_atoms = []
    uniqueA_atoms = []
    uniqueB_atoms = []
    usedB = set()
    # Mapped and uniqueA atoms (A's order)
    for _, rowA in dfA.iterrows():
        idxA = int(rowA["index"])
        if idxA in mapping:
            idxB = mapping[idxA]
            mapped_atoms.append((idxA, rowA["atom_name"], idxA, idxB, "mapped"))
            usedB.add(idxB)
        else:
            uniqueA_atoms.append((idxA, rowA["atom_name"], idxA, None, "uniqueA"))
    # UniqueB atoms (B's order)
    for _, rowB in dfB.iterrows():
        idxB = int(rowB["index"])
        if idxB not in usedB:
            uniqueB_atoms.append((idxB, rowB["atom_name"], None, idxB, "uniqueB"))
    return mapped_atoms + uniqueA_atoms + uniqueB_atoms


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

    # Check if atom names match in order
    for i, (topo_idx, topo_name) in enumerate(topo_atoms):
        if i < len(pdb_atoms):
            pdb_name = pdb_atoms[i]
            if topo_name.strip() != pdb_name.strip():
                print(
                    f"[ERROR] Lambda {lam}: Atom {i + 1} mismatch - Topology: {topo_name}, PDB: {pdb_name}"
                )
                return False

    print(
        f"[INFO] Lambda {lam}: Topology and PDB are synchronized ({len(topo_atoms)} atoms)"
    )
    return True


def hybridize_coords_from_itp_interpolated(
        ligA_mol2, ligB_mol2, hybrid_itp, atom_map_txt, out_pdb, lam
):
    """
    For each lambda, output hybrid coordinates with proper dual topology logic:
    - Lambda 0: Only ligand A atoms have real atom types, ligand B atoms are "DUM"
    - Lambda 1: Only ligand B atoms have real atom types, ligand A atoms are "DUM"
    - 0 < lambda < 1: Only mapped atoms have real atom types, unique atoms are "DUM"
    - Mapped atoms: interpolate between A and B coordinates
    - Unique atoms: use real coordinates if available, otherwise place near mapped centroid
    """
    import random

    coordsA, namesA = parse_mol2_coords(ligA_mol2)
    coordsB, namesB = parse_mol2_coords(ligB_mol2)
    dfA = (
        parse_itp_atoms_full(ligA_mol2.replace(".mol2", ".itp"))
        if ligA_mol2.replace(".mol2", ".itp")
        else None
    )
    dfB = (
        parse_itp_atoms_full(ligB_mol2.replace(".mol2", ".itp"))
        if ligB_mol2.replace(".mol2", ".itp")
        else None
    )
    if dfA is None:
        dfA = pd.DataFrame(
            [{"index": idx, "atom_name": namesA[idx]} for idx in sorted(coordsA.keys())]
        )
    if dfB is None:
        dfB = pd.DataFrame(
            [{"index": idx, "atom_name": namesB[idx]} for idx in sorted(coordsB.keys())]
        )
    mapping = load_atom_map(atom_map_txt)

    # Get lambda-specific atom list to match the topology
    atom_list = build_lambda_atom_list(dfA, dfB, mapping, lam)

    # Compute centroid of mapped atoms (core ligand)
    mapped_coords = []
    for hybrid_idx, atom_name, origA_idx, origB_idx, atom_type in atom_list:
        if atom_type == "mapped":
            coordA = coordsA.get(origA_idx, None)
            coordB = coordsB.get(origB_idx, None)
            if coordA is not None and coordB is not None:
                coord = tuple((1 - lam) * a + lam * b for a, b in zip(coordA, coordB))
                mapped_coords.append(coord)
    if mapped_coords:
        centroid = tuple(np.mean([c[i] for c in mapped_coords]) for i in range(3))
    else:
        centroid = (0.0, 0.0, 0.0)

        pdb_lines = []
    atom_counter = 0
    for hybrid_idx, atom_name, origA_idx, origB_idx, atom_type in atom_list:
        atom_counter += 1
        # Determine atom type based on lambda and atom type
        if atom_type == "mapped":
            # Mapped atoms always have real atom types (they're always present)
            coordA = coordsA.get(origA_idx, (0.0, 0.0, 0.0))
            coordB = coordsB.get(origB_idx, (0.0, 0.0, 0.0))
            coord = tuple((1 - lam) * a + lam * b for a, b in zip(coordA, coordB))
            atom_type_pdb = atom_name  # Use real atom name
        elif atom_type == "uniqueA":
            coord = coordsA.get(origA_idx, None)
            if coord is None:
                # Place dummy atom near centroid
                r = random.uniform(0.1, 0.3)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2 * np.pi)
                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)
                coord = (centroid[0] + dx, centroid[1] + dy, centroid[2] + dz)

            # Determine atom type based on lambda
            if lam == 0:
                atom_type_pdb = atom_name  # Real atom type at lambda 0
            elif lam == 1:
                atom_type_pdb = "DUM"  # Dummy at lambda 1
            else:
                atom_type_pdb = "DUM"  # Dummy at intermediate lambda
        elif atom_type == "uniqueB":
            coord = coordsB.get(origB_idx, None)
            if coord is None:
                # Place dummy atom near centroid
                r = random.uniform(0.1, 0.3)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2 * np.pi)
                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)
                coord = (centroid[0] + dx, centroid[1] + dy, centroid[2] + dz)

            # Determine atom type based on lambda
            if lam == 0:
                atom_type_pdb = "DUM"  # Dummy at lambda 0
            elif lam == 1:
                atom_type_pdb = atom_name  # Real atom type at lambda 1
            else:
                atom_type_pdb = "DUM"  # Dummy at intermediate lambda
        else:
            # Should not occur, but fallback
            r = random.uniform(0.1, 0.3)
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2 * np.pi)
            dx = r * np.sin(theta) * np.cos(phi)
            dy = r * np.sin(theta) * np.sin(phi)
            dz = r * np.cos(theta)
            coord = (centroid[0] + dx, centroid[1] + dy, centroid[2] + dz)
            atom_type_pdb = "DUM"

        pdb_lines.append(
            f"HETATM{atom_counter:5d}  {atom_type_pdb:<4s}LIG     1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n"
        )
    with open(out_pdb, "w") as f:
        for line in pdb_lines:
            f.write(line)
        f.write("END\n")


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
    dummy_bond_params = {"r": "0.0", "k": "0.0"}
    dummy_angle_params = {"r": "0.0", "k": "0.0"}
    dummy_dihedral_params = {"r": "0.0", "k": "0.0"}

    # Convert to hybrid terms using filtered sections for both A and B
    hybrid_bonds = build_hybrid_terms(
        bondsA_filtered,
        bondsB_filtered,
        mapping,
        ["ai", "aj"],
        HybridBond,
        dummy_bond_params,
        dummy_bond_params,
        ["r", "k"],
        ["r", "k"],
    )
    hybrid_angles = build_hybrid_terms(
        anglesA_filtered,
        anglesB_filtered,
        mapping,
        ["ai", "aj", "ak"],
        HybridAngle,
        dummy_angle_params,
        dummy_angle_params,
        ["r", "k"],
        ["r", "k"],
    )
    hybrid_dihedrals = build_hybrid_terms(
        dihedA_filtered,
        dihedB_filtered,
        mapping,
        ["ai", "aj", "ak", "al"],
        HybridDihedral,
        dummy_dihedral_params,
        dummy_dihedral_params,
        ["r", "k"],
        ["r", "k"],
    )

    # Filter out any remaining invalid terms from hybrid terms
    hybrid_bonds = [bond for bond in hybrid_bonds if bond.ai != bond.aj]
    hybrid_angles = [
        angle
        for angle in hybrid_angles
        if angle.ai != angle.aj and angle.aj != angle.ak and angle.ai != angle.ak
    ]
    hybrid_dihedrals = [
        dih
        for dih in hybrid_dihedrals
        if dih.ai != dih.aj
           and dih.ai != dih.ak
           and dih.ai != dih.al
           and dih.aj != dih.ak
           and dih.aj != dih.al
           and dih.ak != dih.al
    ]

    return hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals


# =====================
# CLI
# =====================
def print_help():
    print(
        """
Usage:
  python fep_prep.py mcs ligandA.mol2 ligandB.mol2 atom_map.txt
      Find the maximum common substructure and write atom_map.txt
  python fep_prep.py hybrid_topology ligandA.itp ligandB.itp atom_map.txt
      Generate hybrid .itp files for all lambda windows
  python fep_prep.py hybrid_coords ligandA.mol2 ligandB.mol2 atom_map.txt
      Generate hybridized .pdb files for all lambda windows, each in its own lambda_XX directory
"""
    )


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "mcs":
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
        anglesA = parse_itp_section(
            itpA, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"]
        )
        anglesB = parse_itp_section(
            itpB, "angles", 6, ["ai", "aj", "ak", "funct", "r", "k"]
        )
        dihedA = parse_itp_section(
            itpA, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"]
        )
        dihedB = parse_itp_section(
            itpB, "dihedrals", 7, ["ai", "aj", "ak", "al", "funct", "r", "k"]
        )
        lambdas = np.arange(0, 1.05, 0.05)
        for lam in lambdas:
            lam_str = f"{lam:.2f}"
            lam_dir = f"lambda_{lam_str}"
            import os

            if not os.path.exists(lam_dir):
                os.makedirs(lam_dir)
            outfilename = os.path.join(lam_dir, f"hybrid_lambda_{lam_str}.itp")

            # Create hybrid topology for this lambda
            hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals = (
                create_hybrid_topology_for_lambda(
                    dfA,
                    dfB,
                    bondsA,
                    bondsB,
                    anglesA,
                    anglesB,
                    dihedA,
                    dihedB,
                    mapping,
                    lam,
                )
            )

            write_hybrid_topology(
                outfilename,
                hybrid_atoms,
                hybrid_bonds=hybrid_bonds,
                hybrid_angles=hybrid_angles,
                hybrid_dihedrals=hybrid_dihedrals,
                system_name="LigandA to LigandB Hybrid",
                molecule_name="LIG",
                nmols=1,
            )

            # Create position restraints file for this lambda
            posre_filename = os.path.join(lam_dir, "posre_ligand.itp")
            write_position_restraints_file(posre_filename, hybrid_atoms)

            print(f"Wrote {outfilename}")
            if os.path.exists(posre_filename):
                print(f"Wrote {posre_filename}")
    elif cmd == "hybrid_coords":
        if len(sys.argv) != 5:
            print(
                "Usage: python fep_prep.py hybrid_coords ligandA.mol2 ligandB.mol2 atom_map.txt"
            )
            sys.exit(1)
        ligA_mol2, ligB_mol2, atom_map_txt = sys.argv[2:5]
        lambdas = np.arange(0, 1.05, 0.05)
        for lam in lambdas:
            lam_str = f"{lam:.2f}"
            lam_dir = f"lambda_{lam_str}"
            import os

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

            # Verify synchronization
            verify_hybrid_synchronization(hybrid_itp, out_pdb, lam_str)
    else:
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
