def count_residues_in_gro(gro_path, water_resnames=("SOL",)):
    """
    Parses a GROMACS .gro file to count protein and water residues.

    Parameters:
        gro_path (str): Path to the .gro file.
        water_resnames (tuple): Tuple of residue names considered as water.

    Returns:
        dict: {'protein': int, 'water': int}
    """
    residue_ids = set()
    water_ids = set()

    with open(gro_path, 'r') as f:
        lines = f.readlines()

    # Atom lines are from line 3 to N-2 (last two lines are box vectors)
    for line in lines[2:-1]:
        if len(line) < 20:
            continue  # skip malformed lines

        res_id = int(line[:5].strip())
        res_name = line[5:10].strip()

        if res_name in water_resnames:
            water_ids.add(res_id)
        else:
            residue_ids.add(res_id)

    protein_count = len(residue_ids - water_ids)
    water_count = len(water_ids)

    return protein_count, water_count
