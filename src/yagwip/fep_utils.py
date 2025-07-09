import sys
import itertools
from collections import defaultdict
import sys
import pandas as pd


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
    # Fast checks
    if len(g1.atoms) != len(g2.atoms):
        return False, None
    # Build candidate lists for each atom in g1
    candidates = {}
    for idx1, atom1 in g1.atoms.items():
        candidates[idx1] = [idx2 for idx2, atom2 in g2.atoms.items()
                            if atom1.element == atom2.element and atom1.degree == atom2.degree]
        if not candidates[idx1]:
            return False, None
    # Try all permutations with pruning
    def backtrack(mapping, used2):
        if len(mapping) == len(g1.atoms):
            return True, dict(mapping)
        idx1 = next(i for i in g1.atoms if i not in mapping)
        for idx2 in candidates[idx1]:
            if idx2 in used2:
                continue
            # Check neighbor consistency
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
    # Enumerate all connected subgraphs of a given size
    # Use BFS to grow subgraphs
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
    for size in range(len(g1.atoms), 0, -1):
        subgraphs1 = enumerate_connected_subgraphs(g1, size)
        if not subgraphs1:
            continue
        for atom_indices1 in subgraphs1:
            sg1 = g1.subgraph(atom_indices1)
            # For speed, filter by element counts
            elem_count1 = defaultdict(int)
            for a in sg1.atoms.values():
                elem_count1[a.element] += 1
            # Find candidate subgraphs in g2 with same element counts
            subgraphs2 = [s for s in enumerate_connected_subgraphs(g2, size)
                          if all(
                              sum(g2.atoms[i].element == e for i in s) == c
                              for e, c in elem_count1.items()
                          )]
            for atom_indices2 in subgraphs2:
                sg2 = g2.subgraph(atom_indices2)
                iso, mapping = are_isomorphic(sg1, sg2)
                if iso:
                    return size, mapping, atom_indices1, atom_indices2
    return 0, None, None, None

# --- Utility functions ---
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
            if line.strip() == '':
                continue
            a, b = map(int, line.split())
            mapping[a] = b
    return mapping

def parse_itp_atoms(filename):
    # Returns a DataFrame with columns: index, atom_name, typeA, typeB, mapped
    atoms = []
    with open(filename) as f:
        lines = f.readlines()
    in_atoms = False
    for line in lines:
        if line.strip().startswith('[ atoms ]'):
            in_atoms = True
            continue
        if in_atoms:
            if line.strip().startswith('['):
                break
            if line.strip() == '' or line.strip().startswith(';'):
                continue
            parts = line.split()
            # index type resnr residue atom_name cgnr charge mass typeB chargeB massB
            if len(parts) < 11:
                continue
            idx = int(parts[0])
            atom_name = parts[4]
            typeA = parts[1]
            typeB = parts[8]
            mapped = (typeA != 'DUM' and typeB != 'DUM')
            atoms.append({'index': idx, 'atom_name': atom_name, 'typeA': typeA, 'typeB': typeB, 'mapped': mapped})
    return pd.DataFrame(atoms)

# --- Main hybridization logic ---
def hybridize_coords_from_itp(ligA_mol2, ligB_mol2, hybrid_itp, atom_map_txt, out_pdb):
    coordsA, namesA = parse_mol2_coords(ligA_mol2)
    coordsB, namesB = parse_mol2_coords(ligB_mol2)
    hybrid_atoms = parse_itp_atoms(hybrid_itp)
    mapping = load_atom_map(atom_map_txt)

    pdb_lines = []
    for i, row in hybrid_atoms.iterrows():
        idx = int(row['index'])
        atom_name = row['atom_name']
        mapped = row['mapped']
        if mapped:
            coord = coordsA.get(idx, (0.0, 0.0, 0.0))
        else:
            if row['typeA'] != 'DUM':
                coord = coordsA.get(idx, (0.0, 0.0, 0.0))
            elif row['typeB'] != 'DUM':
                # Find B index corresponding to this atom
                b_index = None
                for k, v in mapping.items():
                    if v == idx:
                        b_index = v
                        break
                if b_index is None:
                    b_index = idx
                coord = coordsB.get(b_index, (0.0, 0.0, 0.0))
            else:
                coord = (0.0, 0.0, 0.0)
        pdb_lines.append(f"ATOM  {i+1:5d} {atom_name:<4s} LIG     1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n")
    with open(out_pdb, 'w') as f:
        for line in pdb_lines:
            f.write(line)
        f.write("END\n")



class HybridAtom:
    def __init__(self, index, atom_name, typeA, typeB, chargeA, chargeB, massA, massB, mapped):
        self.index = index
        self.atom_name = atom_name
        self.typeA = typeA
        self.typeB = typeB
        self.chargeA = chargeA
        self.chargeB = chargeB
        self.massA = massA
        self.massB = massB
        self.mapped = mapped

class HybridBond:
    def __init__(self, ai, aj, funct, parA, parB, mapped):
        self.ai = ai
        self.aj = aj
        self.funct = funct
        self.parA = parA
        self.parB = parB
        self.mapped = mapped

class HybridAngle:
    def __init__(self, ai, aj, ak, funct, parA, parB, mapped):
        self.ai = ai
        self.aj = aj
        self.ak = ak
        self.funct = funct
        self.parA = parA
        self.parB = parB
        self.mapped = mapped

class HybridDihedral:
    def __init__(self, ai, aj, ak, al, funct, parA, parB, mapped):
        self.ai = ai
        self.aj = aj
        self.ak = ak
        self.al = al
        self.funct = funct
        self.parA = parA
        self.parB = parB
        self.mapped = mapped

def parse_itp_section(filename, section, ncols, colnames):
    # Generic parser for [ bonds ], [ angles ], [ dihedrals ]
    with open(filename) as f:
        lines = f.readlines()
    data = []
    in_section = False
    for line in lines:
        if line.strip().startswith(f'[ {section} ]'):
            in_section = True
            continue
        if in_section:
            if line.strip().startswith('['):
                break
            if line.strip() == '' or line.strip().startswith(';'):
                continue
            parts = line.split()
            if len(parts) < ncols:
                continue
            row = {colnames[i]: parts[i] for i in range(ncols)}
            data.append(row)
    return pd.DataFrame(data)

def parse_itp_atoms(filename):
    # Parse [ atoms ] section from .itp file into a DataFrame
    with open(filename) as f:
        lines = f.readlines()
    atoms = []
    in_atoms = False
    for line in lines:
        if line.strip().startswith('[ atoms ]'):
            in_atoms = True
            continue
        if in_atoms:
            if line.strip().startswith('['):
                break
            if line.strip() == '' or line.strip().startswith(';'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            atoms.append({
                'index': int(parts[0]),
                'type': parts[1],
                'resnr': int(parts[2]),
                'residue': parts[3],
                'atom_name': parts[4],
                'cgnr': int(parts[5]),
                'charge': float(parts[6]),
                'mass': float(parts[7]) if len(parts) > 7 else 0.0
            })
    return pd.DataFrame(atoms)

def load_atom_map(filename):
    mapping = {}
    with open(filename) as f:
        for line in f:
            if line.strip() == '':
                continue
            a, b = map(int, line.split())
            mapping[a] = b
    return mapping

def build_hybrid_atoms(dfA, dfB, mapping):
    hybrid_atoms = []
    all_indices = set(dfA['index']).union(dfB['index'])
    for idx in sorted(all_indices):
        inA = idx in dfA['index'].values
        inB = idx in mapping and mapping[idx] in dfB['index'].values
        if inA and inB:
            rowA = dfA[dfA['index'] == idx].iloc[0]
            idxB = mapping[idx]
            rowB = dfB[dfB['index'] == idxB].iloc[0]
            hybrid_atoms.append(HybridAtom(
                index=idx,
                atom_name=rowA['atom_name'],
                typeA=rowA['type'],
                typeB=rowB['type'],
                chargeA=rowA['charge'],
                chargeB=rowB['charge'],
                massA=rowA['mass'],
                massB=rowB['mass'],
                mapped=True
            ))
        elif inA:
            rowA = dfA[dfA['index'] == idx].iloc[0]
            hybrid_atoms.append(HybridAtom(
                index=idx,
                atom_name=rowA['atom_name'],
                typeA=rowA['type'],
                typeB='DUM',
                chargeA=rowA['charge'],
                chargeB=0.0,
                massA=rowA['mass'],
                massB=0.0,
                mapped=False
            ))
        else:
            idxB = None
            for k, v in mapping.items():
                if v == idx:
                    idxB = v
                    break
            if idxB is not None:
                rowB = dfB[dfB['index'] == idxB].iloc[0]
            else:
                rowB = dfB[dfB['index'] == idx].iloc[0]
            hybrid_atoms.append(HybridAtom(
                index=idx,
                atom_name=rowB['atom_name'],
                typeA='DUM',
                typeB=rowB['type'],
                chargeA=0.0,
                chargeB=rowB['charge'],
                massA=0.0,
                massB=rowB['mass'],
                mapped=False
            ))
    return hybrid_atoms

def build_hybrid_terms(dfA, dfB, mapping, keycols, HybridClass, dummyA, dummyB):
    # Merge/interpolate bonds, angles, dihedrals
    # keycols: columns to match (e.g., ['ai','aj'] for bonds)
    # dummyA/B: dummy values for unique terms
    termsA = dfA.copy()
    termsB = dfB.copy()
    # Convert indices in B to A's mapping for matching
    inv_map = {v: k for k, v in mapping.items()}
    def map_indices(row, inv=False):
        for col in keycols:
            idx = int(row[col])
            if inv:
                if idx in inv_map:
                    row[col] = inv_map[idx]
            else:
                if idx in mapping:
                    row[col] = mapping[idx]
        return row
    termsB = termsB.apply(lambda row: map_indices(row, inv=True), axis=1)
    # Merge on keycols
    merged = pd.merge(termsA, termsB, on=keycols, how='outer', suffixes=('A','B'), indicator=True)
    hybrid_terms = []
    for _, row in merged.iterrows():
        mapped = row['_merge'] == 'both'
        valsA = [row.get(f'{c}A', dummyA[c]) for c in dummyA.keys()]
        valsB = [row.get(f'{c}B', dummyB[c]) for c in dummyB.keys()]
        indices = [int(row[c]) for c in keycols]
        # Robust funct handling for all hybrid term classes
        functA = row.get('functA')
        functB = row.get('functB')
        funct = functA if pd.notnull(functA) else functB
        if pd.isnull(funct):
            funct = 1
        funct = int(funct)
        # Unpack all required parameters for each class
        if HybridClass == HybridBond:
            hybrid_terms.append(HybridBond(*indices, funct, *valsA, *valsB, mapped))
        elif HybridClass == HybridAngle:
            hybrid_terms.append(HybridAngle(*indices, funct, *valsA, *valsB, mapped))
        elif HybridClass == HybridDihedral:
            hybrid_terms.append(HybridDihedral(*indices, funct, *valsA, *valsB, mapped))
    return hybrid_terms

def print_hybrid_topology(
    hybrid_atoms, hybrid_bonds=None, hybrid_pairs=None, hybrid_angles=None, hybrid_dihedrals=None,
    system_name="Hybrid System", molecule_name="HybridMol", nmols=1, forcefield="gromos43a1.ff/forcefield.itp"
):
    print(f'; Include force field parameters')
    print(f'#include "{forcefield}"\n')
    print('[ moleculetype ]')
    print('; Name            nrexcl')
    print(f'{molecule_name:<18}3\n')
    print('[ atoms ]')
    print('; nr type resnr residue atom cgnr  charge    mass  typeB chargeB  massB')
    for atom in hybrid_atoms:
        print(f'{atom.index:4d} {atom.typeA:6s} {1:4d} {"RES":6s} {atom.atom_name:4s} {1:4d} '
              f'{atom.chargeA:8.4f} {atom.massA:7.3f} {atom.typeB:6s} {atom.chargeB:8.4f} {atom.massB:7.3f}')
    print()
    if hybrid_bonds is not None:
        print('[ bonds ]')
        print(';  ai    aj funct    par_A  par_B')
        for bond in hybrid_bonds:
            print(f'{bond.ai:5d} {bond.aj:5d} {bond.funct:5d} {bond.parA:7s} {bond.parB:7s}')
        print()
    if hybrid_pairs is not None:
        print('[ pairs ]')
        print(';  ai    aj funct')
        for pair in hybrid_pairs:
            print(f'{pair.ai:5d} {pair.aj:5d} {pair.funct:5d}')
        print()
    if hybrid_angles is not None:
        print('[ angles ]')
        print(';  ai    aj    ak funct    par_A   par_B')
        for angle in hybrid_angles:
            print(f'{angle.ai:5d} {angle.aj:5d} {angle.ak:5d} {angle.funct:5d} {angle.parA:7s} {angle.parB:7s}')
        print()
    if hybrid_dihedrals is not None:
        print('[ dihedrals ]')
        print(';  ai    aj    ak    al funct    par_A   par_B')
        for dih in hybrid_dihedrals:
            print(f'{dih.ai:5d} {dih.aj:5d} {dih.ak:5d} {dih.al:5d} {dih.funct:5d} {dih.parA:7s} {dih.parB:7s}')
        print()
    print('[ system ]')
    print('; Name')
    print(system_name)
    print()
    print('[ molecules ]')
    print('; Compound        #mols')
    print(f'{molecule_name:<18}{nmols}')
    print()

def write_hybrid_topology(
    filename,
    hybrid_atoms, hybrid_bonds=None, hybrid_pairs=None, hybrid_angles=None, hybrid_dihedrals=None,
    system_name="Hybrid System", molecule_name="HybridMol", nmols=1, forcefield="gromos43a1.ff/forcefield.itp"
):
    with open(filename, 'w') as f:
        f.write(f'; Include force field parameters\n')
        f.write(f'#include "{forcefield}"\n\n')
        f.write('[ moleculetype ]\n')
        f.write('; Name            nrexcl\n')
        f.write(f'{molecule_name:<18}3\n\n')
        f.write('[ atoms ]\n')
        f.write('; nr type resnr residue atom cgnr  charge    mass  typeB chargeB  massB\n')
        for atom in hybrid_atoms:
            f.write(f'{atom.index:4d} {atom.typeA:6s} {1:4d} {"RES":6s} {atom.atom_name:4s} {1:4d} '
                    f'{atom.chargeA:8.4f} {atom.massA:7.3f} {atom.typeB:6s} {atom.chargeB:8.4f} {atom.massB:7.3f}\n')
        f.write('\n')
        if hybrid_bonds is not None:
            f.write('[ bonds ]\n')
            f.write(';  ai    aj funct    par_A  par_B\n')
            for bond in hybrid_bonds:
                f.write(f'{bond.ai:5d} {bond.aj:5d} {bond.funct:5d} {bond.parA:7s} {bond.parB:7s}\n')
            f.write('\n')
        if hybrid_pairs is not None:
            f.write('[ pairs ]\n')
            f.write(';  ai    aj funct\n')
            for pair in hybrid_pairs:
                f.write(f'{pair.ai:5d} {pair.aj:5d} {pair.funct:5d}\n')
            f.write('\n')
        if hybrid_angles is not None:
            f.write('[ angles ]\n')
            f.write(';  ai    aj    ak funct    par_A   par_B\n')
            for angle in hybrid_angles:
                f.write(f'{angle.ai:5d} {angle.aj:5d} {angle.ak:5d} {angle.funct:5d} {angle.parA:7s} {angle.parB:7s}\n')
            f.write('\n')
        if hybrid_dihedrals is not None:
            f.write('[ dihedrals ]\n')
            f.write(';  ai    aj    ak    al funct    par_A   par_B\n')
            for dih in hybrid_dihedrals:
                f.write(f'{dih.ai:5d} {dih.aj:5d} {dih.ak:5d} {dih.al:5d} {dih.funct:5d} {dih.parA:7s} {dih.parB:7s}\n')
            f.write('\n')
        f.write('[ system ]\n')
        f.write('; Name\n')
        f.write(f'{system_name}\n\n')
        f.write('[ molecules ]\n')
        f.write('; Compound        #mols\n')
        f.write(f'{molecule_name:<18}{nmols}\n\n')

def main():
    if len(sys.argv) != 4:
        print("Usage: python hybrid_topology.py ligandA.itp ligandB.itp atom_map.txt")
        sys.exit(1)
    itpA, itpB, mapfile = sys.argv[1:4]
    dfA = parse_itp_atoms(itpA)
    dfB = parse_itp_atoms(itpB)
    mapping = load_atom_map(mapfile)
    hybrid_atoms = build_hybrid_atoms(dfA, dfB, mapping)
    # Parse and merge bonds, angles, dihedrals
    bondsA = parse_itp_section(itpA, 'bonds', 4, ['ai','aj','funct','parA'])
    bondsB = parse_itp_section(itpB, 'bonds', 4, ['ai','aj','funct','parB'])
    hybrid_bonds = build_hybrid_terms(bondsA, bondsB, mapping, ['ai','aj'], HybridBond, {'parA':'gb_0'}, {'parB':'gb_0'})
    anglesA = parse_itp_section(itpA, 'angles', 5, ['ai','aj','ak','funct','parA'])
    anglesB = parse_itp_section(itpB, 'angles', 5, ['ai','aj','ak','funct','parB'])
    hybrid_angles = build_hybrid_terms(anglesA, anglesB, mapping, ['ai','aj','ak'], HybridAngle, {'parA':'ga_0'}, {'parB':'ga_0'})
    dihedA = parse_itp_section(itpA, 'dihedrals', 6, ['ai','aj','ak','al','funct','parA'])
    dihedB = parse_itp_section(itpB, 'dihedrals', 6, ['ai','aj','ak','al','funct','parB'])
    hybrid_dihedrals = build_hybrid_terms(dihedA, dihedB, mapping, ['ai','aj','ak','al'], HybridDihedral, {'parA':'gd_0'}, {'parB':'gd_0'})
    # For each lambda, print the hybrid topology
    lambdas = np.arange(0, 1.05, 0.05)
    for lam in lambdas:
        # Interpolate atom charges/masses
        for atom in hybrid_atoms:
            atom.chargeA = (1-lam)*atom.chargeA + lam*atom.chargeB
            atom.massA = (1-lam)*atom.massA + lam*atom.massB
        print(f"\n; ===== Lambda = {lam:.2f} =====\n")
        print_hybrid_topology(
            hybrid_atoms,
            hybrid_bonds=hybrid_bonds,
            hybrid_angles=hybrid_angles,
            hybrid_dihedrals=hybrid_dihedrals,
            system_name="LigandA to LigandB Hybrid",
            molecule_name="PropPent",
            nmols=200,
            forcefield="gromos43a1.ff/forcefield.itp"
        )

    outfilename = f"hybrid_lambda_{lam:.2f}.itp"
    write_hybrid_topology(
        outfilename,
        hybrid_atoms,
        hybrid_bonds=hybrid_bonds,
        hybrid_angles=hybrid_angles,
        hybrid_dihedrals=hybrid_dihedrals,
        system_name="LigandA to LigandB Hybrid",
        molecule_name="PropPent",
        nmols=200,
        forcefield="gromos43a1.ff/forcefield.itp"
    )