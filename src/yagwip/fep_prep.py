import os
import argparse
import numpy as np
from shutil import copyfile

# --- Kabsch algorithm implementation ---
def kabsch(P, Q):
    # P and Q are Nx3 matrices
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    return U

def load_pdb_coords(pdb_file):
    coords = []
    atom_lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                atom_lines.append(line)
    return np.array(coords), atom_lines

def write_aligned_pdb(atom_lines, coords, out_file):
    with open(out_file, 'w') as f:
        for line, (x, y, z) in zip(atom_lines, coords):
            newline = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            f.write(newline)

def align_ligandB_to_A(ligA_pdb, ligB_pdb, out_pdb):
    P, _ = load_pdb_coords(ligA_pdb)
    Q, atom_lines = load_pdb_coords(ligB_pdb)
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    U = kabsch(Q_cent, P_cent)
    Q_aligned = np.dot(Q_cent, U)
    Q_aligned += P.mean(axis=0)
    write_aligned_pdb(atom_lines, Q_aligned, out_pdb)

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description='FEP prep: align ligands, generate water/complex systems.')
    parser.add_argument('--ligA_pdb', required=True)
    parser.add_argument('--ligA_itp', required=True)
    parser.add_argument('--ligB_pdb', required=True)
    parser.add_argument('--ligB_itp', required=True)
    parser.add_argument('--protein_gro', required=True)
    args = parser.parse_args()

    # Step 1: Align ligandB to ligandA
    aligned_ligB_pdb = 'ligandB_aligned.pdb'
    align_ligandB_to_A(args.ligA_pdb, args.ligB_pdb, aligned_ligB_pdb)

    # Step 2: Create output directories
    out_dirs = ['A_water', 'A_complex', 'B_water', 'B_complex']
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    # Step 3: Copy input files to respective directories
    copyfile(args.ligA_pdb, os.path.join('A_water', 'ligandA.pdb'))
    copyfile(args.ligA_itp, os.path.join('A_water', 'ligandA.itp'))
    copyfile(args.ligA_pdb, os.path.join('A_complex', 'ligandA.pdb'))
    copyfile(args.ligA_itp, os.path.join('A_complex', 'ligandA.itp'))
    copyfile(aligned_ligB_pdb, os.path.join('B_water', 'ligandB.pdb'))
    copyfile(args.ligB_itp, os.path.join('B_water', 'ligandB.itp'))
    copyfile(aligned_ligB_pdb, os.path.join('B_complex', 'ligandB.pdb'))
    copyfile(args.ligB_itp, os.path.join('B_complex', 'ligandB.itp'))
    copyfile(args.protein_gro, os.path.join('A_complex', 'protein.gro'))
    copyfile(args.protein_gro, os.path.join('B_complex', 'protein.gro'))

    # Step 4: Placeholder for YAGWIP/GROMACS steps
    print('TODO: Run pdb2gmx, solvate, genion for each system using YAGWIP utilities.')
    # Example:
    # run_pdb2gmx(os.path.join('A_water', 'ligandA.pdb'), ...)
    # run_solvate(...)
    # run_genion(...)

if __name__ == '__main__':
    main()
