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


def find_mcs(g1, g2, target_size=10):
    """
    Find Maximum Common Substructure (MCS) between two molecular graphs.
    Conservative approach: stop as soon as we find a connected MCS of target_size atoms.

    Args:
        g1: MolGraph for ligand A
        g2: MolGraph for ligand B
        target_size: Target MCS size (default: 10 atoms)

    Returns:
        Tuple of (mcs_size, mapping, atom_indices1, atom_indices2)
    """
    # Always use the smaller graph for subgraph enumeration
    if len(g1.atoms) > len(g2.atoms):
        g1, g2 = g2, g1

    # Start from target_size and work down to minimum size
    for size in range(target_size, 2, -1):  # Start from target_size, need at least 3 for alignment
        print(f"Searching for MCS of size {size}...")
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
                    # Verify MCS connectivity - ensure atoms form a continuous structure
                    if verify_mcs_connectivity(sg1, sg2, mapping):
                        print(f"Found valid MCS of size {size}!")
                        return size, mapping, atom_indices1, atom_indices2

    # If we didn't find target_size, try smaller sizes
    for size in range(target_size - 1, 2, -1):
        print(f"Target size not found, trying size {size}...")
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
                    # Verify MCS connectivity - ensure atoms form a continuous structure
                    if verify_mcs_connectivity(sg1, sg2, mapping):
                        print(f"Found valid MCS of size {size}!")
                        return size, mapping, atom_indices1, atom_indices2

    return 0, None, None, None


def verify_mcs_connectivity(sg1, sg2, mapping):
    """
    Verify that MCS atoms form a continuous, connected structure.

    Args:
        sg1: Subgraph from ligand A
        sg2: Subgraph from ligand B
        mapping: Atom mapping between the two subgraphs

    Returns:
        True if MCS is continuous, False if disconnected fragments exist
    """
    # Check connectivity in both subgraphs
    if not is_graph_connected(sg1) or not is_graph_connected(sg2):
        return False

    # Additional check: ensure no isolated atoms or small disconnected fragments
    # Calculate the diameter of the MCS (longest shortest path)
    diameter1 = calculate_graph_diameter(sg1)
    diameter2 = calculate_graph_diameter(sg2)

    # If diameter is very large relative to number of atoms, it might indicate disconnected fragments
    # A reasonable threshold: diameter should be less than number of atoms
    if diameter1 > len(sg1.atoms) or diameter2 > len(sg2.atoms):
        return False

    # Check for any atoms that are too far from the center of mass
    if not check_atom_distances_from_center(sg1) or not check_atom_distances_from_center(sg2):
        return False

    return True


def is_graph_connected(graph):
    """
    Check if a graph is connected using BFS.

    Args:
        graph: MolGraph object

    Returns:
        True if graph is connected, False otherwise
    """
    if not graph.atoms:
        return False

    # Start BFS from the first atom
    start_atom = next(iter(graph.atoms.keys()))
    visited = set()
    queue = [start_atom]

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            # Add all unvisited neighbors
            for neighbor in graph.atoms[current].neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    # Check if all atoms were visited
    return len(visited) == len(graph.atoms)


def calculate_graph_diameter(graph):
    """
    Calculate the diameter of a graph (longest shortest path between any two vertices).

    Args:
        graph: MolGraph object

    Returns:
        Diameter of the graph
    """
    if not graph.atoms:
        return 0

    max_diameter = 0

    # For each pair of atoms, calculate shortest path length
    atom_indices = list(graph.atoms.keys())
    for i, start in enumerate(atom_indices):
        for end in atom_indices[i + 1:]:
            path_length = shortest_path_length(graph, start, end)
            if path_length > max_diameter:
                max_diameter = path_length

    return max_diameter


def shortest_path_length(graph, start, end):
    """
    Calculate shortest path length between two atoms using BFS.

    Args:
        graph: MolGraph object
        start: Starting atom index
        end: Ending atom index

    Returns:
        Length of shortest path, or float('inf') if no path exists
    """
    if start == end:
        return 0

    visited = set()
    queue = [(start, 0)]  # (atom, distance)

    while queue:
        current, distance = queue.pop(0)
        if current == end:
            return distance

        if current not in visited:
            visited.add(current)
            for neighbor in graph.atoms[current].neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    return float('inf')  # No path found


def check_atom_distances_from_center(graph, max_distance_factor=3.0):
    """
    Check that no atoms are too far from the center of the MCS.

    Args:
        graph: MolGraph object
        max_distance_factor: Maximum allowed distance as factor of graph diameter

    Returns:
        True if all atoms are reasonably close to center, False otherwise
    """
    if not graph.atoms:
        return True

    # Calculate center of mass (using atom indices as proxy for spatial position)
    center = sum(graph.atoms.keys()) / len(graph.atoms)

    # Calculate maximum distance from center
    max_distance = max(abs(atom_idx - center) for atom_idx in graph.atoms.keys())

    # Calculate graph diameter
    diameter = calculate_graph_diameter(graph)

    # If max distance is too large relative to diameter, it might indicate disconnected fragments
    if diameter > 0 and max_distance > max_distance_factor * diameter:
        return False

    return True


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
        def pdb_to_gro(pdb_file, gro_file, use_aligned_coords=False):
            """Convert PDB to GRO format, optionally using aligned coordinates"""
            with open(pdb_file, 'r') as f:
                lines = f.readlines()

            # Count atoms (skip REMARK and END lines)
            atom_lines = [line for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
            num_atoms = len(atom_lines)

            with open(gro_file, 'w') as f:
                # Write header (title line)
                f.write("Hybrid structure for FEP (Extended Form)\n")
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

        # Convert PDB to GRO using the hybrid PDB files (which already have aligned coordinates)
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

            # Get topol.top template from templates directory
            topol_template_path = files("templates").joinpath("topol.top")
            with open(str(topol_template_path), 'r', encoding='utf-8') as f:
                topol_template = f.read()
            print(f"Using topol.top template from {topol_template_path}")

        # Write topol.top files to A_to_B and B_to_A level
        with open(os.path.join(ligand_a_to_b, 'topol.top'), 'w') as f:
            f.write(topol_template)
        with open(os.path.join(ligand_b_to_a, 'topol.top'), 'w') as f:
            f.write(topol_template)

        # Copy FEP MDP files from templates to each A_to_B and B_to_A directory
        fep_mdp_files = ["em_fep.mdp", "nvt_fep.mdp", "npt_fep.mdp", "production_fep.mdp"]
        fep_directories = [ligand_a_to_b, ligand_b_to_a, protein_a_to_b, protein_b_to_a]

        for fep_dir in fep_directories:
            for mdp_file in fep_mdp_files:
                mdp_template_path = files("templates").joinpath(mdp_file)
                if mdp_template_path.is_file():
                    dest_path = os.path.join(fep_dir, mdp_file)
                    with open(str(mdp_template_path), 'r', encoding='utf-8') as f:
                        content = f.read()
                    with open(dest_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Copied {mdp_file} to {fep_dir}")
                else:
                    print(f"Warning: {mdp_file} template not found at {mdp_template_path}")

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

    # Parse exclusions and pairs from ITP files
    exclusionsA = parse_itp_exclusions(ligA_itp)
    exclusionsB = parse_itp_exclusions(ligB_itp)
    pairsA = parse_itp_pairs(ligA_itp)
    pairsB = parse_itp_pairs(ligB_itp)

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
    hybrid_bonds = create_hybrid_bonds(atomsA, atomsB, mapping, hybrid_atoms, ligA_itp, ligB_itp)

    # Create hybrid angles
    hybrid_angles = create_hybrid_angles(atomsA, atomsB, mapping, hybrid_atoms, ligA_itp, ligB_itp)

    # Create hybrid dihedrals
    hybrid_dihedrals = create_hybrid_dihedrals(atomsA, atomsB, mapping, hybrid_atoms, ligA_itp, ligB_itp)

    # Filter exclusions and pairs based on best practices
    hybrid_exclusions = filter_exclusions_for_hybrid(exclusionsA, exclusionsB, hybrid_atoms, coordsA, coordsB_aligned)
    hybrid_pairs = filter_pairs_for_hybrid(pairsA, pairsB, hybrid_atoms, coordsA, coordsB_aligned)

    # Write hybrid topology
    write_hybrid_itp(out_itp, hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, hybrid_exclusions,
                     hybrid_pairs)

    # Write hybrid PDB files for both states
    write_hybrid_pdb(out_pdbA, hybrid_atoms, coordsA, coordsB_aligned, mapping, state='A')
    write_hybrid_pdb(out_pdbB, hybrid_atoms, coordsA, coordsB_aligned, mapping, state='B')

    print(f"Created hybrid topology with {len(hybrid_atoms)} atoms")
    print(f"  - {len([a for a in hybrid_atoms if a['mapped']])} mapped atoms")
    print(f"  - {len([a for a in hybrid_atoms if not a['mapped'] and a['origA_idx']])} unique A atoms")
    print(f"  - {len([a for a in hybrid_atoms if not a['mapped'] and a['origB_idx']])} unique B atoms")
    print(f"  - {len(hybrid_exclusions)} filtered exclusions")
    print(f"  - {len(hybrid_pairs)} filtered pairs")


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


def parse_itp_bonds(itp_file):
    """Parse bond section from ITP file."""
    bonds = []
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_bonds = False
    for line in lines:
        if line.strip() == '[ bonds ]':
            in_bonds = True
            continue
        elif in_bonds and line.strip().startswith('['):
            break
        elif in_bonds and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 3:
                bond = {
                    'ai': int(parts[0]),
                    'aj': int(parts[1]),
                    'funct': int(parts[2])
                }
                # Add bond parameters if present
                if len(parts) >= 5:
                    bond['length'] = float(parts[3])
                    bond['force'] = float(parts[4])
                bonds.append(bond)

    return bonds


def parse_itp_angles(itp_file):
    """Parse angle section from ITP file."""
    angles = []
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_angles = False
    for line in lines:
        if line.strip() == '[ angles ]':
            in_angles = True
            continue
        elif in_angles and line.strip().startswith('['):
            break
        elif in_angles and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 4:
                angle = {
                    'ai': int(parts[0]),
                    'aj': int(parts[1]),
                    'ak': int(parts[2]),
                    'funct': int(parts[3])
                }
                # Add angle parameters if present
                if len(parts) >= 6:
                    angle['theta'] = float(parts[4])
                    angle['fc'] = float(parts[5])
                angles.append(angle)

    return angles


def parse_itp_dihedrals(itp_file):
    """Parse dihedral section from ITP file."""
    dihedrals = []
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_dihedrals = False
    for line in lines:
        if line.strip() == '[ dihedrals ]':
            in_dihedrals = True
            continue
        elif in_dihedrals and line.strip().startswith('['):
            break
        elif in_dihedrals and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 5:
                dihedral = {
                    'ai': int(parts[0]),
                    'aj': int(parts[1]),
                    'ak': int(parts[2]),
                    'al': int(parts[3]),
                    'funct': int(parts[4])
                }
                # Add dihedral parameters if present
                if len(parts) >= 8:
                    dihedral['phi'] = float(parts[5])
                    dihedral['fc'] = float(parts[6])
                    dihedral['mult'] = int(parts[7])
                dihedrals.append(dihedral)

    return dihedrals


def create_hybrid_bonds(atomsA, atomsB, mapping, hybrid_atoms, ligA_itp, ligB_itp):
    """Create hybrid bonds using the merge strategy with complete connectivity."""
    # Parse bond sections from both ITP files
    bondsA = parse_itp_bonds(ligA_itp)
    bondsB = parse_itp_bonds(ligB_itp)

    # Create mapping from original atom indices to hybrid indices
    hybrid_bonds = []

    # Track which hybrid atoms are connected
    connected_atoms = set()

    # Process bonds from ligand A
    for bondA in bondsA:
        ai_A, aj_A = bondA['ai'], bondA['aj']

        # Find corresponding hybrid atom indices
        ai_hybrid = None
        aj_hybrid = None

        for atom in hybrid_atoms:
            if atom['origA_idx'] == ai_A:
                ai_hybrid = atom['index']
            if atom['origA_idx'] == aj_A:
                aj_hybrid = atom['index']

        if ai_hybrid is not None and aj_hybrid is not None:
            # Check if this bond exists in ligand B
            bond_exists_in_B = False
            bondB_params = None

            for bondB in bondsB:
                if (bondB['ai'] == ai_A and bondB['aj'] == aj_A) or \
                        (bondB['ai'] == aj_A and bondB['aj'] == ai_A):
                    bond_exists_in_B = True
                    bondB_params = bondB
                    break

            hybrid_bond = {
                'ai': ai_hybrid,
                'aj': aj_hybrid,
                'funct': bondA['funct']
            }

            if bond_exists_in_B and bondB_params is not None:
                # Bond exists in both A and B
                hybrid_bond['lengthA'] = bondA.get('length', 0.14)
                hybrid_bond['forceA'] = bondA.get('force', 50000)
                hybrid_bond['lengthB'] = bondB_params.get('length', 0.14)
                hybrid_bond['forceB'] = bondB_params.get('force', 50000)
            else:
                # Bond exists only in A
                hybrid_bond['lengthA'] = bondA.get('length', 0.14)
                hybrid_bond['forceA'] = bondA.get('force', 50000)
                hybrid_bond['lengthB'] = 0.1  # Dummy value
                hybrid_bond['forceB'] = 0.0  # Zero force

            hybrid_bonds.append(hybrid_bond)
            connected_atoms.add(ai_hybrid)
            connected_atoms.add(aj_hybrid)

    # Process bonds from ligand B (for bonds that don't exist in A)
    for bondB in bondsB:
        ai_B, aj_B = bondB['ai'], bondB['aj']

        # Check if this bond was already processed from ligand A
        already_processed = False
        for bondA in bondsA:
            if (bondA['ai'] == ai_B and bondA['aj'] == aj_B) or \
                    (bondA['ai'] == aj_B and bondA['aj'] == ai_B):
                already_processed = True
                break

        if not already_processed:
            # Find corresponding hybrid atom indices
            ai_hybrid = None
            aj_hybrid = None

            for atom in hybrid_atoms:
                if atom['origB_idx'] == ai_B:
                    ai_hybrid = atom['index']
                if atom['origB_idx'] == aj_B:
                    aj_hybrid = atom['index']

            if ai_hybrid is not None and aj_hybrid is not None:
                # Bond exists only in B
                hybrid_bond = {
                    'ai': ai_hybrid,
                    'aj': aj_hybrid,
                    'funct': bondB['funct'],
                    'lengthA': 0.1,  # Dummy value
                    'forceA': 0.0,  # Zero force
                    'lengthB': bondB.get('length', 0.14),
                    'forceB': bondB.get('force', 50000)
                }
                hybrid_bonds.append(hybrid_bond)
                connected_atoms.add(ai_hybrid)
                connected_atoms.add(aj_hybrid)

    # Ensure all dummy atoms are connected to real atoms
    # Find dummy atoms that aren't connected
    dummy_atoms = [atom for atom in hybrid_atoms if not atom['mapped']]
    mapped_atoms = [atom for atom in hybrid_atoms if atom['mapped']]

    for dummy_atom in dummy_atoms:
        if dummy_atom['index'] not in connected_atoms:
            # Find the closest mapped atom to connect this dummy atom
            closest_mapped = None
            min_distance = float('inf')

            # For simplicity, connect to the first mapped atom
            # In practice, you might want to use spatial coordinates
            if mapped_atoms:
                closest_mapped = mapped_atoms[0]

                # Create a connecting bond with zero force in the appropriate state
                if dummy_atom['origA_idx'] is not None:
                    # This is a unique A atom (dummy in state B)
                    hybrid_bond = {
                        'ai': dummy_atom['index'],
                        'aj': closest_mapped['index'],
                        'funct': 1,
                        'lengthA': 0.14,  # Normal bond in state A
                        'forceA': 50000,
                        'lengthB': 0.1,  # Dummy bond in state B
                        'forceB': 0.0
                    }
                else:
                    # This is a unique B atom (dummy in state A)
                    hybrid_bond = {
                        'ai': dummy_atom['index'],
                        'aj': closest_mapped['index'],
                        'funct': 1,
                        'lengthA': 0.1,  # Dummy bond in state A
                        'forceA': 0.0,
                        'lengthB': 0.14,  # Normal bond in state B
                        'forceB': 50000
                    }

                hybrid_bonds.append(hybrid_bond)
                connected_atoms.add(dummy_atom['index'])

    return hybrid_bonds


def create_hybrid_angles(atomsA, atomsB, mapping, hybrid_atoms, ligA_itp, ligB_itp):
    """Create hybrid angles using the merge strategy with complete connectivity."""
    # Parse angle sections from both ITP files
    anglesA = parse_itp_angles(ligA_itp)
    anglesB = parse_itp_angles(ligB_itp)

    hybrid_angles = []

    # Process angles from ligand A
    for angleA in anglesA:
        ai_A, aj_A, ak_A = angleA['ai'], angleA['aj'], angleA['ak']

        # Find corresponding hybrid atom indices
        ai_hybrid = None
        aj_hybrid = None
        ak_hybrid = None

        for atom in hybrid_atoms:
            if atom['origA_idx'] == ai_A:
                ai_hybrid = atom['index']
            if atom['origA_idx'] == aj_A:
                aj_hybrid = atom['index']
            if atom['origA_idx'] == ak_A:
                ak_hybrid = atom['index']

        if ai_hybrid is not None and aj_hybrid is not None and ak_hybrid is not None:
            # Check if this angle exists in ligand B
            angle_exists_in_B = False
            angleB_params = None

            for angleB in anglesB:
                if (angleB['ai'] == ai_A and angleB['aj'] == aj_A and angleB['ak'] == ak_A):
                    angle_exists_in_B = True
                    angleB_params = angleB
                    break

            hybrid_angle = {
                'ai': ai_hybrid,
                'aj': aj_hybrid,
                'ak': ak_hybrid,
                'funct': angleA['funct']
            }

            if angle_exists_in_B and angleB_params is not None:
                # Angle exists in both A and B
                hybrid_angle['thetaA'] = angleA.get('theta', 109.5)
                hybrid_angle['fcA'] = angleA.get('fc', 520)
                hybrid_angle['thetaB'] = angleB_params.get('theta', 109.5)
                hybrid_angle['fcB'] = angleB_params.get('fc', 520)
            else:
                # Angle exists only in A
                hybrid_angle['thetaA'] = angleA.get('theta', 109.5)
                hybrid_angle['fcA'] = angleA.get('fc', 520)
                hybrid_angle['thetaB'] = 120.0  # Dummy value
                hybrid_angle['fcB'] = 0.0  # Zero force

            hybrid_angles.append(hybrid_angle)

    # Process angles from ligand B (for angles that don't exist in A)
    for angleB in anglesB:
        ai_B, aj_B, ak_B = angleB['ai'], angleB['aj'], angleB['ak']

        # Check if this angle was already processed from ligand A
        already_processed = False
        for angleA in anglesA:
            if (angleA['ai'] == ai_B and angleA['aj'] == aj_B and angleA['ak'] == ak_B):
                already_processed = True
                break

        if not already_processed:
            # Find corresponding hybrid atom indices
            ai_hybrid = None
            aj_hybrid = None
            ak_hybrid = None

            for atom in hybrid_atoms:
                if atom['origB_idx'] == ai_B:
                    ai_hybrid = atom['index']
                if atom['origB_idx'] == aj_B:
                    aj_hybrid = atom['index']
                if atom['origB_idx'] == ak_B:
                    ak_hybrid = atom['index']

            if ai_hybrid is not None and aj_hybrid is not None and ak_hybrid is not None:
                # Angle exists only in B
                hybrid_angle = {
                    'ai': ai_hybrid,
                    'aj': aj_hybrid,
                    'ak': ak_hybrid,
                    'funct': angleB['funct'],
                    'thetaA': 120.0,  # Dummy value
                    'fcA': 0.0,  # Zero force
                    'thetaB': angleB.get('theta', 109.5),
                    'fcB': angleB.get('fc', 520)
                }
                hybrid_angles.append(hybrid_angle)

    # Add missing angles to ensure complete connectivity
    # This ensures that all dummy atoms have proper angle terms
    hybrid_angles = add_missing_angles_for_connectivity(hybrid_atoms, hybrid_angles)

    return hybrid_angles


def add_missing_angles_for_connectivity(hybrid_atoms, existing_angles):
    """Add missing angles to ensure all dummy atoms have proper connectivity."""
    # Find dummy atoms
    dummy_atoms = [atom for atom in hybrid_atoms if not atom['mapped']]
    mapped_atoms = [atom for atom in hybrid_atoms if atom['mapped']]

    if not dummy_atoms or not mapped_atoms:
        return existing_angles

    # For each dummy atom, ensure it has angle terms with mapped atoms
    for dummy_atom in dummy_atoms:
        # Find existing angles involving this dummy atom
        dummy_angles = [angle for angle in existing_angles
                        if dummy_atom['index'] in [angle['ai'], angle['aj'], angle['ak']]]

        # If dummy atom doesn't have enough angle terms, add some
        if len(dummy_angles) < 2:  # Most atoms should have at least 2 angle terms
            # Find two mapped atoms to create angles with
            mapped_neighbors = []
            for mapped_atom in mapped_atoms:
                if len(mapped_neighbors) >= 2:
                    break
                # Create angle: mapped1 - dummy - mapped2
                if len(mapped_neighbors) == 0:
                    mapped_neighbors.append(mapped_atom)
                elif len(mapped_neighbors) == 1:
                    # Check if this angle already exists
                    angle_exists = False
                    for angle in existing_angles:
                        if (angle['ai'] == mapped_neighbors[0]['index'] and
                                angle['aj'] == dummy_atom['index'] and
                                angle['ak'] == mapped_atom['index']):
                            angle_exists = True
                            break
                        elif (angle['ai'] == mapped_atom['index'] and
                              angle['aj'] == dummy_atom['index'] and
                              angle['ak'] == mapped_neighbors[0]['index']):
                            angle_exists = True
                            break

                    if not angle_exists:
                        mapped_neighbors.append(mapped_atom)

            # Add missing angles
            if len(mapped_neighbors) == 2:
                # Create angle: mapped1 - dummy - mapped2
                if dummy_atom['origA_idx'] is not None:
                    # This is a unique A atom (dummy in state B)
                    new_angle = {
                        'ai': mapped_neighbors[0]['index'],
                        'aj': dummy_atom['index'],
                        'ak': mapped_neighbors[1]['index'],
                        'funct': 1,
                        'thetaA': 109.5,  # Normal angle in state A
                        'fcA': 520,
                        'thetaB': 120.0,  # Dummy angle in state B
                        'fcB': 0.0
                    }
                else:
                    # This is a unique B atom (dummy in state A)
                    new_angle = {
                        'ai': mapped_neighbors[0]['index'],
                        'aj': dummy_atom['index'],
                        'ak': mapped_neighbors[1]['index'],
                        'funct': 1,
                        'thetaA': 120.0,  # Dummy angle in state A
                        'fcA': 0.0,
                        'thetaB': 109.5,  # Normal angle in state B
                        'fcB': 520
                    }

                existing_angles.append(new_angle)

    return existing_angles


def create_hybrid_dihedrals(atomsA, atomsB, mapping, hybrid_atoms, ligA_itp, ligB_itp):
    """Create hybrid dihedrals using the merge strategy with complete connectivity."""
    # Parse dihedral sections from both ITP files
    dihedralsA = parse_itp_dihedrals(ligA_itp)
    dihedralsB = parse_itp_dihedrals(ligB_itp)

    hybrid_dihedrals = []

    # Process dihedrals from ligand A
    for dihedralA in dihedralsA:
        ai_A, aj_A, ak_A, al_A = dihedralA['ai'], dihedralA['aj'], dihedralA['ak'], dihedralA['al']

        # Find corresponding hybrid atom indices
        ai_hybrid = None
        aj_hybrid = None
        ak_hybrid = None
        al_hybrid = None

        for atom in hybrid_atoms:
            if atom['origA_idx'] == ai_A:
                ai_hybrid = atom['index']
            if atom['origA_idx'] == aj_A:
                aj_hybrid = atom['index']
            if atom['origA_idx'] == ak_A:
                ak_hybrid = atom['index']
            if atom['origA_idx'] == al_A:
                al_hybrid = atom['index']

        if all(x is not None for x in [ai_hybrid, aj_hybrid, ak_hybrid, al_hybrid]):
            # Check if this dihedral exists in ligand B
            dihedral_exists_in_B = False
            dihedralB_params = None

            for dihedralB in dihedralsB:
                if (dihedralB['ai'] == ai_A and dihedralB['aj'] == aj_A and
                        dihedralB['ak'] == ak_A and dihedralB['al'] == al_A):
                    dihedral_exists_in_B = True
                    dihedralB_params = dihedralB
                    break

            hybrid_dihedral = {
                'ai': ai_hybrid,
                'aj': aj_hybrid,
                'ak': ak_hybrid,
                'al': al_hybrid,
                'funct': dihedralA['funct']
            }

            if dihedral_exists_in_B and dihedralB_params is not None:
                # Dihedral exists in both A and B - use consistent multiplicity
                mult_A = dihedralA.get('mult', 3)
                mult_B = dihedralB_params.get('mult', 3)

                # Use the higher multiplicity to ensure proper sampling
                consistent_mult = max(mult_A, mult_B)

                hybrid_dihedral['phiA'] = dihedralA.get('phi', 180.0)
                hybrid_dihedral['fcA'] = dihedralA.get('fc', 2.0)
                hybrid_dihedral['mult'] = consistent_mult  # Single multiplicity for both states
                hybrid_dihedral['phiB'] = dihedralB_params.get('phi', 180.0)
                hybrid_dihedral['fcB'] = dihedralB_params.get('fc', 2.0)
            else:
                # Dihedral exists only in A
                hybrid_dihedral['phiA'] = dihedralA.get('phi', 180.0)
                hybrid_dihedral['fcA'] = dihedralA.get('fc', 2.0)
                hybrid_dihedral['mult'] = dihedralA.get('mult', 3)  # Single multiplicity
                hybrid_dihedral['phiB'] = 180.0  # Dummy value
                hybrid_dihedral['fcB'] = 0.0  # Zero force

            hybrid_dihedrals.append(hybrid_dihedral)

    # Process dihedrals from ligand B (for dihedrals that don't exist in A)
    for dihedralB in dihedralsB:
        ai_B, aj_B, ak_B, al_B = dihedralB['ai'], dihedralB['aj'], dihedralB['ak'], dihedralB['al']

        # Check if this dihedral was already processed from ligand A
        already_processed = False
        for dihedralA in dihedralsA:
            if (dihedralA['ai'] == ai_B and dihedralA['aj'] == aj_B and
                    dihedralA['ak'] == ak_B and dihedralA['al'] == al_B):
                already_processed = True
                break

        if not already_processed:
            # Find corresponding hybrid atom indices
            ai_hybrid = None
            aj_hybrid = None
            ak_hybrid = None
            al_hybrid = None

            for atom in hybrid_atoms:
                if atom['origB_idx'] == ai_B:
                    ai_hybrid = atom['index']
                if atom['origB_idx'] == aj_B:
                    aj_hybrid = atom['index']
                if atom['origB_idx'] == ak_B:
                    ak_hybrid = atom['index']
                if atom['origB_idx'] == al_B:
                    al_hybrid = atom['index']

            if all(x is not None for x in [ai_hybrid, aj_hybrid, ak_hybrid, al_hybrid]):
                # Dihedral exists only in B
                hybrid_dihedral = {
                    'ai': ai_hybrid,
                    'aj': aj_hybrid,
                    'ak': ak_hybrid,
                    'al': al_hybrid,
                    'funct': dihedralB['funct'],
                    'phiA': 180.0,  # Dummy value
                    'fcA': 0.0,  # Zero force
                    'mult': dihedralB.get('mult', 3),  # Single multiplicity
                    'phiB': dihedralB.get('phi', 180.0),
                    'fcB': dihedralB.get('fc', 2.0)
                }
                hybrid_dihedrals.append(hybrid_dihedral)

    # Add missing dihedrals to ensure complete connectivity
    # This ensures that all dummy atoms have proper dihedral terms
    hybrid_dihedrals = add_missing_dihedrals_for_connectivity(hybrid_atoms, hybrid_dihedrals)

    return hybrid_dihedrals


def add_missing_dihedrals_for_connectivity(hybrid_atoms, existing_dihedrals):
    """Add missing dihedrals to ensure all dummy atoms have proper connectivity."""
    # Find dummy atoms
    dummy_atoms = [atom for atom in hybrid_atoms if not atom['mapped']]
    mapped_atoms = [atom for atom in hybrid_atoms if atom['mapped']]

    if not dummy_atoms or len(mapped_atoms) < 3:
        return existing_dihedrals

    # For each dummy atom, ensure it has dihedral terms with mapped atoms
    for dummy_atom in dummy_atoms:
        # Find existing dihedrals involving this dummy atom
        dummy_dihedrals = [dihedral for dihedral in existing_dihedrals
                           if dummy_atom['index'] in [dihedral['ai'], dihedral['aj'],
                                                      dihedral['ak'], dihedral['al']]]

        # If dummy atom doesn't have enough dihedral terms, add some
        if len(dummy_dihedrals) < 1:  # Most atoms should have at least 1 dihedral term
            # Find three mapped atoms to create dihedrals with
            mapped_neighbors = []
            for mapped_atom in mapped_atoms:
                if len(mapped_neighbors) >= 3:
                    break
                mapped_neighbors.append(mapped_atom)

            # Add missing dihedrals
            if len(mapped_neighbors) == 3:
                # Create dihedral: mapped1 - mapped2 - dummy - mapped3
                if dummy_atom['origA_idx'] is not None:
                    # This is a unique A atom (dummy in state B)
                    new_dihedral = {
                        'ai': mapped_neighbors[0]['index'],
                        'aj': mapped_neighbors[1]['index'],
                        'ak': dummy_atom['index'],
                        'al': mapped_neighbors[2]['index'],
                        'funct': 1,
                        'phiA': 180.0,  # Normal dihedral in state A
                        'fcA': 2.0,
                        'mult': 3,  # Consistent multiplicity
                        'phiB': 180.0,  # Dummy dihedral in state B
                        'fcB': 0.0
                    }
                else:
                    # This is a unique B atom (dummy in state A)
                    new_dihedral = {
                        'ai': mapped_neighbors[0]['index'],
                        'aj': mapped_neighbors[1]['index'],
                        'ak': dummy_atom['index'],
                        'al': mapped_neighbors[2]['index'],
                        'funct': 1,
                        'phiA': 180.0,  # Dummy dihedral in state A
                        'fcA': 0.0,
                        'mult': 3,  # Consistent multiplicity
                        'phiB': 180.0,  # Normal dihedral in state B
                        'fcB': 2.0
                    }

                existing_dihedrals.append(new_dihedral)

    return existing_dihedrals


def write_hybrid_itp(out_file, hybrid_atoms, hybrid_bonds, hybrid_angles, hybrid_dihedrals, hybrid_exclusions,
                     hybrid_pairs):
    """Write hybrid topology file with dual-state parameters."""
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

        # Add bond section with dual-state parameters
        if hybrid_bonds:
            f.write("[ bonds ]\n")
            f.write("; ai    aj funct  lengthA  forceA   lengthB  forceB\n")
            for bond in hybrid_bonds:
                if 'lengthA' in bond and 'forceA' in bond and 'lengthB' in bond and 'forceB' in bond:
                    f.write(f"{bond['ai']:5d} {bond['aj']:5d} {bond['funct']:5d} "
                            f"{bond['lengthA']:8.3f} {bond['forceA']:8.1f} "
                            f"{bond['lengthB']:8.3f} {bond['forceB']:8.1f}\n")
                else:
                    f.write(f"{bond['ai']:5d} {bond['aj']:5d} {bond['funct']:5d}\n")
            f.write("\n")

        # Add angle section with dual-state parameters
        if hybrid_angles:
            f.write("[ angles ]\n")
            f.write("; ai    aj    ak funct  thetaA  fcA   thetaB  fcB\n")
            for angle in hybrid_angles:
                if 'thetaA' in angle and 'fcA' in angle and 'thetaB' in angle and 'fcB' in angle:
                    f.write(f"{angle['ai']:5d} {angle['aj']:5d} {angle['ak']:5d} {angle['funct']:5d} "
                            f"{angle['thetaA']:8.2f} {angle['fcA']:6.1f} "
                            f"{angle['thetaB']:8.2f} {angle['fcB']:6.1f}\n")
                else:
                    f.write(f"{angle['ai']:5d} {angle['aj']:5d} {angle['ak']:5d} {angle['funct']:5d}\n")
            f.write("\n")

        # Add dihedral section with dual-state parameters
        if hybrid_dihedrals:
            f.write("[ dihedrals ]\n")
            f.write("; ai    aj    ak    al funct  phiA  fcA  mult   phiB  fcB  mult\n")
            for dih in hybrid_dihedrals:
                if 'phiA' in dih and 'fcA' in dih and 'mult' in dih and 'phiB' in dih and 'fcB' in dih:
                    f.write(f"{dih['ai']:5d} {dih['aj']:5d} {dih['ak']:5d} {dih['al']:5d} {dih['funct']:5d} "
                            f"{dih['phiA']:6.1f} {dih['fcA']:5.1f} {dih['mult']:5d} "
                            f"{dih['phiB']:6.1f} {dih['fcB']:5.1f} {dih['mult']:5d}\n")
                else:
                    f.write(f"{dih['ai']:5d} {dih['aj']:5d} {dih['ak']:5d} {dih['al']:5d} {dih['funct']:5d}\n")
            f.write("\n")

        # Add position restraints for dummy atoms
        dummy_atoms = [atom for atom in hybrid_atoms if not atom['mapped']]
        if dummy_atoms:
            f.write("[ position_restraints ]\n")
            f.write("; ai  funct  fx      fy      fz\n")
            for atom in dummy_atoms:
                f.write(f"{atom['index']:4d}     1   1000    1000    1000\n")
            f.write("\n")

        # Add exclusions section
        if hybrid_exclusions:
            f.write("[ exclusions ]\n")
            f.write("; ai    aj\n")
            for excl in hybrid_exclusions:
                f.write(f"{excl['ai']:5d} {excl['aj']:5d}\n")
            f.write("\n")

        # Add pairs section
        if hybrid_pairs:
            f.write("[ pairs ]\n")
            f.write("; ai    aj funct  param\n")
            for pair in hybrid_pairs:
                if 'funct' in pair and 'param' in pair:
                    f.write(f"{pair['ai']:5d} {pair['aj']:5d} {pair['funct']:5d} {pair['param']:8.3f}\n")
                else:
                    f.write(f"{pair['ai']:5d} {pair['aj']:5d} {pair['funct']:5d}\n")
            f.write("\n")


def write_hybrid_pdb(out_file, hybrid_atoms, coordsA, coordsB_aligned, mapping, state='A'):
    """
    Write hybrid PDB file for specified state using sphere-packing algorithm.

    Sphere-packing strategy:
    - All atoms are surrounded by 1.2 nm radius spheres
    - Non-MCS atoms are positioned so their spheres touch other ligand atom spheres
    - All atoms must be connected to MCS through network of touching spheres
    - Minimum distance constraint: 1.2 nm between all atoms

    Args:
        out_file: Output PDB file path
        hybrid_atoms: List of hybrid atom dictionaries
        coordsA: Coordinates for ligand A (original)
        coordsB_aligned: Aligned coordinates for ligand B (after Kabsch alignment)
        mapping: Atom mapping dictionary
        state: 'A' or 'B' to determine which state to write
    """
    with open(out_file, 'w') as f:
        f.write("REMARK Hybrid structure for FEP (Sphere-Packing)\n")
        f.write(f"REMARK State: {state}\n")
        f.write(f"REMARK All atoms separated by >= 1.2 nm, connected to MCS via touching spheres\n")

        # Generate sphere-packed coordinates
        sphere_coords = generate_sphere_packed_coordinates(hybrid_atoms, coordsA, coordsB_aligned, mapping, state)

        for atom in hybrid_atoms:
            atom_idx = atom['index']
            if atom_idx in sphere_coords:
                x, y, z = sphere_coords[atom_idx]
                atom_name = atom['name'] if atom['mapped'] else 'DUM'
                f.write(f"HETATM{atom_idx:5d}  {atom_name:<4s}LIG     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
            else:
                # Fallback to centroid if sphere packing failed
                x, y, z = get_centroid_of_mapped_atoms(hybrid_atoms, coordsA, coordsB_aligned)
                atom_name = 'DUM'
                f.write(f"HETATM{atom_idx:5d}  {atom_name:<4s}LIG     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

        f.write("END\n")


def generate_sphere_packed_coordinates(hybrid_atoms, coordsA, coordsB_aligned, mapping, state, sphere_radius=1.2):
    """
    Generate sphere-packed coordinates ensuring minimum distance and MCS connectivity.

    Args:
        hybrid_atoms: List of hybrid atom dictionaries
        coordsA: Coordinates for ligand A
        coordsB_aligned: Aligned coordinates for ligand B
        mapping: Atom mapping dictionary
        state: 'A' or 'B'
        sphere_radius: Radius of spheres (default 1.2 nm)

    Returns:
        Dictionary mapping atom index to (x, y, z) coordinates
    """
    import math
    import random

    # Initialize coordinates dictionary
    sphere_coords = {}

    # Identify MCS atoms (mapped atoms)
    mcs_atoms = [atom for atom in hybrid_atoms if atom['mapped']]
    non_mcs_atoms = [atom for atom in hybrid_atoms if not atom['mapped']]

    # Step 1: Position MCS atoms using original coordinates
    for atom in mcs_atoms:
        if state == 'A' and atom['origA_idx'] in coordsA:
            sphere_coords[atom['index']] = coordsA[atom['origA_idx']]
        elif state == 'B' and atom['origB_idx'] in coordsB_aligned:
            sphere_coords[atom['index']] = coordsB_aligned[atom['origB_idx']]
        else:
            # Fallback to centroid
            sphere_coords[atom['index']] = get_centroid_of_mapped_atoms(hybrid_atoms, coordsA, coordsB_aligned)

    # Step 2: Position non-MCS atoms using sphere-packing algorithm
    for atom in non_mcs_atoms:
        if state == 'A' and atom['origB_idx'] in coordsB_aligned:
            # For state A, non-MCS atoms are unique to B - use aligned B coordinates as starting point
            initial_coord = coordsB_aligned[atom['origB_idx']]
        elif state == 'B' and atom['origA_idx'] in coordsA:
            # For state B, non-MCS atoms are unique to A - use A coordinates as starting point
            initial_coord = coordsA[atom['origA_idx']]
        else:
            # Fallback to centroid
            initial_coord = get_centroid_of_mapped_atoms(hybrid_atoms, coordsA, coordsB_aligned)

        # Find optimal position that satisfies distance and connectivity constraints
        optimal_coord = find_optimal_sphere_position(
            atom['index'], initial_coord, sphere_coords, sphere_radius
        )
        sphere_coords[atom['index']] = optimal_coord

    return sphere_coords


def find_optimal_sphere_position(atom_idx, initial_coord, existing_coords, sphere_radius, max_iterations=1000):
    """
    Find optimal position for an atom that satisfies sphere-packing constraints.

    Args:
        atom_idx: Index of the atom to position
        initial_coord: Initial coordinate guess
        existing_coords: Dictionary of existing atom coordinates
        sphere_radius: Radius of spheres
        max_iterations: Maximum iterations for optimization

    Returns:
        Optimal (x, y, z) coordinates
    """
    import math
    import random

    def distance(coord1, coord2):
        """Calculate Euclidean distance between two coordinates."""
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        dz = coord1[2] - coord2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def is_valid_position(coord, existing_coords, sphere_radius):
        """Check if position satisfies distance constraints."""
        for existing_idx, existing_coord in existing_coords.items():
            if existing_idx != atom_idx:
                dist = distance(coord, existing_coord)
                if dist < sphere_radius:
                    return False
        return True

    def find_touching_sphere_position(coord, existing_coords, sphere_radius):
        """Find position where sphere touches another sphere."""
        best_coord = coord
        min_distance = float('inf')

        for existing_idx, existing_coord in existing_coords.items():
            if existing_idx != atom_idx:
                dist = distance(coord, existing_coord)
                if dist < min_distance:
                    min_distance = dist
                    # Position sphere to touch the closest existing sphere
                    if dist > 0:
                        # Normalize direction vector
                        direction = [
                            (coord[0] - existing_coord[0]) / dist,
                            (coord[1] - existing_coord[1]) / dist,
                            (coord[2] - existing_coord[2]) / dist
                        ]
                        # Position at exactly sphere_radius distance
                        best_coord = [
                            existing_coord[0] + direction[0] * sphere_radius,
                            existing_coord[1] + direction[1] * sphere_radius,
                            existing_coord[2] + direction[2] * sphere_radius
                        ]

        return best_coord

    # Start with initial coordinate
    current_coord = list(initial_coord)

    # Iterative optimization
    for iteration in range(max_iterations):
        # Check if current position is valid
        if is_valid_position(current_coord, existing_coords, sphere_radius):
            return tuple(current_coord)

        # Find position that touches another sphere
        touching_coord = find_touching_sphere_position(current_coord, existing_coords, sphere_radius)

        # Add small random perturbation to avoid getting stuck
        perturbation = 0.1 * sphere_radius
        perturbed_coord = [
            touching_coord[0] + random.uniform(-perturbation, perturbation),
            touching_coord[1] + random.uniform(-perturbation, perturbation),
            touching_coord[2] + random.uniform(-perturbation, perturbation)
        ]

        # If perturbed position is valid, use it
        if is_valid_position(perturbed_coord, existing_coords, sphere_radius):
            return tuple(perturbed_coord)

        current_coord = perturbed_coord

    # If optimization failed, return the best touching position found
    return find_touching_sphere_position(initial_coord, existing_coords, sphere_radius)


def verify_sphere_packing_connectivity(hybrid_atoms, sphere_coords, sphere_radius=1.2):
    """
    Verify that all atoms are connected to MCS through network of touching spheres.

    Args:
        hybrid_atoms: List of hybrid atom dictionaries
        sphere_coords: Dictionary of atom coordinates
        sphere_radius: Radius of spheres

    Returns:
        True if all atoms are connected, False otherwise
    """
    import math

    def distance(coord1, coord2):
        """Calculate Euclidean distance between two coordinates."""
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        dz = coord1[2] - coord2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def are_spheres_touching(coord1, coord2, sphere_radius):
        """Check if two spheres are touching (distance <= 2*radius)."""
        dist = distance(coord1, coord2)
        return dist <= 2.0 * sphere_radius

    # Build connectivity graph
    connectivity = {}
    mcs_atoms = set()

    for atom in hybrid_atoms:
        atom_idx = atom['index']
        connectivity[atom_idx] = []

        if atom['mapped']:
            mcs_atoms.add(atom_idx)

        # Find all atoms that this atom's sphere touches
        for other_atom in hybrid_atoms:
            other_idx = other_atom['index']
            if other_idx != atom_idx:
                if (atom_idx in sphere_coords and other_idx in sphere_coords and
                        are_spheres_touching(sphere_coords[atom_idx], sphere_coords[other_idx], sphere_radius)):
                    connectivity[atom_idx].append(other_idx)

    # Check connectivity using BFS from MCS atoms
    visited = set()
    queue = list(mcs_atoms)

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            for neighbor in connectivity[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

    # All atoms should be reachable from MCS
    all_atoms = {atom['index'] for atom in hybrid_atoms}
    return visited == all_atoms


def get_centroid_of_mapped_atoms(hybrid_atoms, coordsA, coordsB_aligned):
    """
    Calculate centroid of mapped atoms for fallback positioning.

    Args:
        hybrid_atoms: List of hybrid atom dictionaries
        coordsA: Coordinates for ligand A
        coordsB_aligned: Aligned coordinates for ligand B

    Returns:
        Tuple of (x, y, z) coordinates for centroid
    """
    mapped_coords = []

    for atom in hybrid_atoms:
        if atom['mapped']:
            if atom['origA_idx'] in coordsA:
                mapped_coords.append(coordsA[atom['origA_idx']])
            elif atom['origB_idx'] in coordsB_aligned:
                mapped_coords.append(coordsB_aligned[atom['origB_idx']])

    if not mapped_coords:
        return (0.0, 0.0, 0.0)

    # Calculate centroid
    centroid_x = sum(coord[0] for coord in mapped_coords) / len(mapped_coords)
    centroid_y = sum(coord[1] for coord in mapped_coords) / len(mapped_coords)
    centroid_z = sum(coord[2] for coord in mapped_coords) / len(mapped_coords)

    return (centroid_x, centroid_y, centroid_z)


def parse_itp_exclusions(itp_file):
    """Parse exclusion section from ITP file."""
    exclusions = []
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_exclusions = False
    for line in lines:
        if line.strip() == '[ exclusions ]':
            in_exclusions = True
            continue
        elif in_exclusions and line.strip().startswith('['):
            break
        elif in_exclusions and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 2:
                exclusion = {
                    'ai': int(parts[0]),
                    'aj': int(parts[1])
                }
                exclusions.append(exclusion)

    return exclusions


def parse_itp_pairs(itp_file):
    """Parse pairs section from ITP file."""
    pairs = []
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_pairs = False
    for line in lines:
        if line.strip() == '[ pairs ]':
            in_pairs = True
            continue
        elif in_pairs and line.strip().startswith('['):
            break
        elif in_pairs and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 2:
                # Check if the line contains valid numeric data
                try:
                    pair = {
                        'ai': int(parts[0]),
                        'aj': int(parts[1])
                    }
                    # Add pair parameters if present and valid
                    if len(parts) >= 4:
                        try:
                            pair['funct'] = int(parts[2])
                            pair['param'] = float(parts[3])
                        except (ValueError, IndexError):
                            # If parameters can't be parsed, skip them
                            pass
                    pairs.append(pair)
                except (ValueError, IndexError):
                    # Skip lines that can't be parsed as valid pairs
                    continue

    return pairs


def filter_exclusions_for_hybrid(exclusionsA, exclusionsB, hybrid_atoms, coordsA, coordsB_aligned, rlist=1.1):
    """
    Filter exclusions based on hybrid topology best practices.

    Rules:
    1. Only keep exclusions for atoms that exist in both states
    2. Filter by distance (within rlist) in both lambda states
    3. Exclude dummy atoms from exclusions
    """
    import math

    def calculate_distance(coord1, coord2):
        """Calculate Euclidean distance between two 3D coordinates."""
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        dz = coord1[2] - coord2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # Create mapping from original atom indices to hybrid indices
    orig_to_hybrid_A = {}
    orig_to_hybrid_B = {}

    for atom in hybrid_atoms:
        if atom['origA_idx'] is not None:
            orig_to_hybrid_A[atom['origA_idx']] = atom['index']
        if atom['origB_idx'] is not None:
            orig_to_hybrid_B[atom['origB_idx']] = atom['index']

    filtered_exclusions = []

    # Process exclusions from ligand A
    for excl in exclusionsA:
        ai_A, aj_A = excl['ai'], excl['aj']

        # Check if both atoms exist in hybrid topology
        if ai_A in orig_to_hybrid_A and aj_A in orig_to_hybrid_A:
            ai_hybrid = orig_to_hybrid_A[ai_A]
            aj_hybrid = orig_to_hybrid_A[aj_A]

            # Check if both atoms are mapped (not dummy)
            ai_atom = next((a for a in hybrid_atoms if a['index'] == ai_hybrid), None)
            aj_atom = next((a for a in hybrid_atoms if a['index'] == aj_hybrid), None)

            if ai_atom and aj_atom and ai_atom['mapped'] and aj_atom['mapped']:
                # Check distance in both states
                if (ai_A in coordsA and aj_A in coordsA and
                        ai_A in coordsB_aligned and aj_A in coordsB_aligned):

                    dist_A = calculate_distance(coordsA[ai_A], coordsA[aj_A])
                    dist_B = calculate_distance(coordsB_aligned[ai_A], coordsB_aligned[aj_A])

                    # Only keep if close in both states
                    if dist_A < rlist and dist_B < rlist:
                        filtered_exclusions.append({
                            'ai': ai_hybrid,
                            'aj': aj_hybrid
                        })

    # Process exclusions from ligand B (avoid duplicates)
    for excl in exclusionsB:
        ai_B, aj_B = excl['ai'], excl['aj']

        # Check if both atoms exist in hybrid topology
        if ai_B in orig_to_hybrid_B and aj_B in orig_to_hybrid_B:
            ai_hybrid = orig_to_hybrid_B[ai_B]
            aj_hybrid = orig_to_hybrid_B[aj_B]

            # Check if both atoms are mapped (not dummy)
            ai_atom = next((a for a in hybrid_atoms if a['index'] == ai_hybrid), None)
            aj_atom = next((a for a in hybrid_atoms if a['index'] == aj_hybrid), None)

            if ai_atom and aj_atom and ai_atom['mapped'] and aj_atom['mapped']:
                # Check if this exclusion was already added from ligand A
                already_exists = any(
                    (e['ai'] == ai_hybrid and e['aj'] == aj_hybrid) or
                    (e['ai'] == aj_hybrid and e['aj'] == ai_hybrid)
                    for e in filtered_exclusions
                )

                if not already_exists:
                    # Check distance in both states
                    if (ai_B in coordsA and aj_B in coordsA and
                            ai_B in coordsB_aligned and aj_B in coordsB_aligned):

                        dist_A = calculate_distance(coordsA[ai_B], coordsA[aj_B])
                        dist_B = calculate_distance(coordsB_aligned[ai_B], coordsB_aligned[aj_B])

                        # Only keep if close in both states
                        if dist_A < rlist and dist_B < rlist:
                            filtered_exclusions.append({
                                'ai': ai_hybrid,
                                'aj': aj_hybrid
                            })

    return filtered_exclusions


def filter_pairs_for_hybrid(pairsA, pairsB, hybrid_atoms, coordsA, coordsB_aligned, rlist=1.1):
    """
    Filter pairs based on hybrid topology best practices.

    Rules:
    1. Only keep pairs for atoms that exist in the relevant state
    2. Filter by distance (within rlist)
    3. Exclude dummy atoms from pairs
    """
    import math

    def calculate_distance(coord1, coord2):
        """Calculate Euclidean distance between two 3D coordinates."""
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        dz = coord1[2] - coord2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # Create mapping from original atom indices to hybrid indices
    orig_to_hybrid_A = {}
    orig_to_hybrid_B = {}

    for atom in hybrid_atoms:
        if atom['origA_idx'] is not None:
            orig_to_hybrid_A[atom['origA_idx']] = atom['index']
        if atom['origB_idx'] is not None:
            orig_to_hybrid_B[atom['origB_idx']] = atom['index']

    filtered_pairs = []

    # Process pairs from ligand A
    for pair in pairsA:
        ai_A, aj_A = pair['ai'], pair['aj']

        # Check if both atoms exist in hybrid topology
        if ai_A in orig_to_hybrid_A and aj_A in orig_to_hybrid_A:
            ai_hybrid = orig_to_hybrid_A[ai_A]
            aj_hybrid = orig_to_hybrid_A[aj_A]

            # Check if both atoms are mapped (not dummy)
            ai_atom = next((a for a in hybrid_atoms if a['index'] == ai_hybrid), None)
            aj_atom = next((a for a in hybrid_atoms if a['index'] == aj_hybrid), None)

            if ai_atom and aj_atom and ai_atom['mapped'] and aj_atom['mapped']:
                # Check distance in state A
                if ai_A in coordsA and aj_A in coordsA:
                    dist_A = calculate_distance(coordsA[ai_A], coordsA[aj_A])

                    # Only keep if close in state A
                    if dist_A < rlist:
                        filtered_pair = {
                            'ai': ai_hybrid,
                            'aj': aj_hybrid
                        }
                        if 'funct' in pair:
                            filtered_pair['funct'] = pair['funct']
                        if 'param' in pair:
                            filtered_pair['param'] = pair['param']
                        filtered_pairs.append(filtered_pair)

    # Process pairs from ligand B (avoid duplicates)
    for pair in pairsB:
        ai_B, aj_B = pair['ai'], pair['aj']

        # Check if both atoms exist in hybrid topology
        if ai_B in orig_to_hybrid_B and aj_B in orig_to_hybrid_B:
            ai_hybrid = orig_to_hybrid_B[ai_B]
            aj_hybrid = orig_to_hybrid_B[aj_B]

            # Check if both atoms are mapped (not dummy)
            ai_atom = next((a for a in hybrid_atoms if a['index'] == ai_hybrid), None)
            aj_atom = next((a for a in hybrid_atoms if a['index'] == aj_hybrid), None)

            if ai_atom and aj_atom and ai_atom['mapped'] and aj_atom['mapped']:
                # Check if this pair was already added from ligand A
                already_exists = any(
                    (p['ai'] == ai_hybrid and p['aj'] == aj_hybrid) or
                    (p['ai'] == aj_hybrid and p['aj'] == ai_hybrid)
                    for p in filtered_pairs
                )

                if not already_exists:
                    # Check distance in state B
                    if ai_B in coordsB_aligned and aj_B in coordsB_aligned:
                        dist_B = calculate_distance(coordsB_aligned[ai_B], coordsB_aligned[aj_B])

                        # Only keep if close in state B
                        if dist_B < rlist:
                            filtered_pair = {
                                'ai': ai_hybrid,
                                'aj': aj_hybrid
                            }
                            if 'funct' in pair:
                                filtered_pair['funct'] = pair['funct']
                            if 'param' in pair:
                                filtered_pair['param'] = pair['param']
                            filtered_pairs.append(filtered_pair)

    return filtered_pairs


# def validate_mcs_quality(mapping, gA, gB, min_mcs_size=3):
#     """
#     Validate the quality of the found MCS to ensure it forms a continuous structure.
#
#     Args:
#         mapping: Atom mapping dictionary
#         gA: MolGraph for ligand A
#         gB: MolGraph for ligand B
#         min_mcs_size: Minimum required MCS size
#
#     Returns:
#         Tuple of (is_valid, reason) where is_valid is boolean and reason is string
#     """
#     if not mapping or len(mapping) < min_mcs_size:
#         return False, f"MCS too small: {len(mapping) if mapping else 0} atoms (minimum {min_mcs_size})"
#
#     # Extract MCS subgraphs
#     mcs_atoms_A = set(mapping.keys())
#     mcs_atoms_B = set(mapping.values())
#
#     # Create MCS subgraphs
#     mcs_gA = gA.subgraph(mcs_atoms_A)
#     mcs_gB = gB.subgraph(mcs_atoms_B)
#
#     # Check connectivity
#     if not is_graph_connected(mcs_gA):
#         return False, "MCS atoms in ligand A are not connected"
#
#     if not is_graph_connected(mcs_gB):
#         return False, "MCS atoms in ligand B are not connected"
#
#     # Check diameter constraints
#     diameter_A = calculate_graph_diameter(mcs_gA)
#     diameter_B = calculate_graph_diameter(mcs_gB)
#
#     if diameter_A > len(mcs_atoms_A):
#         return False, f"MCS diameter in ligand A too large: {diameter_A} > {len(mcs_atoms_A)}"
#
#     if diameter_B > len(mcs_atoms_B):
#         return False, f"MCS diameter in ligand B too large: {diameter_B} > {len(mcs_atoms_B)}"
#
#     # Check for reasonable spatial distribution
#     if not check_atom_distances_from_center(mcs_gA):
#         return False, "MCS atoms in ligand A are too far apart (disconnected fragments)"
#
#     if not check_atom_distances_from_center(mcs_gB):
#         return False, "MCS atoms in ligand B are too far apart (disconnected fragments)"
#
#     return True, f"Valid MCS: {len(mapping)} atoms, diameter A={diameter_A}, B={diameter_B}"


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
    parser.add_argument('--min_mcs_size', type=int, default=3,
                        help='Minimum MCS size (default: 3)')
    parser.add_argument('--target_mcs_size', type=int, default=10,
                        help='Target MCS size - stop searching when found (default: 10)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.ligA_mol2))

    # 1. Find MCS and write atom_map.txt
    print(f"Searching for MCS with target size {args.target_mcs_size}...")
    gA = MolGraph.from_mol2(args.ligA_mol2)
    gB = MolGraph.from_mol2(args.ligB_mol2)
    mcs_size, mapping, atom_indicesA, atom_indicesB = find_mcs(gA, gB, args.target_mcs_size)

    if mcs_size < args.min_mcs_size or mapping is None:
        raise RuntimeError(
            f"Could not find sufficient MCS for alignment (need at least {args.min_mcs_size} atoms, found {mcs_size})")

    print(f"MCS found: {mcs_size} atoms (target was {args.target_mcs_size})")

    # Validate MCS quality
    # is_valid, reason = validate_mcs_quality(mapping, gA, gB, args.min_mcs_size)
    # if not is_valid:
    #     raise RuntimeError(f"MCS quality check failed: {reason}")

    # print(f"MCS validation passed: {reason}")

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
