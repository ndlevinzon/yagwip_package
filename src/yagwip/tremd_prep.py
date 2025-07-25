"""
Temperature Replica Exchange Molecular Dynamics (T-REMD) Preparation Module

This module provides tools for preparing Temperature Replica Exchange Molecular Dynamics (T-REMD)
simulations. It implements the temperature ladder generation algorithm based on the work of
Patriksson and van der Spoel (2008) to optimize exchange probabilities between temperature replicas.

Key Features:
- Automatic residue counting from GROMACS GRO files
- Temperature ladder generation with optimized exchange probabilities
- Support for different constraint models and force field configurations
- Interactive command-line interface for parameter input

Algorithm:
The temperature ladder is generated using an iterative approach that:
1. Calculates system degrees of freedom based on constraints
2. Estimates exchange probability between adjacent temperature replicas
3. Adjusts temperature spacing to achieve desired exchange probability
4. Uses empirical coefficients from Patriksson and van der Spoel (2008)

Usage:
    python tremd_prep.py complex.solvated.ions.gro
    # Interactive prompts for Tlow, Thigh, and exchange probability

Reference:
    Alexandra Patriksson and David van der Spoel, "A temperature predictor for parallel tempering
    simulations", Phys. Chem. Chem. Phys., 10 pp. 2073-2077 (2008)
    http://dx.doi.org/10.1039/b716554d

Author: YAGWIP Development Team
"""

# === Standard Library Imports ===
import os
import sys
import math

# === Third-Party Imports ===
import numpy as np


# === T-REMD Utilities ===
# Based on http://dx.doi.org/10.1039/b716554d


def count_residues_in_gro(gro_path, water_resnames=("SOL",)):
    """
    Parse a GROMACS .gro file to count protein and water residues.

    This function analyzes the GRO file structure to identify and count different types
    of residues, which is essential for calculating system degrees of freedom in T-REMD.

    Args:
        gro_path (str): Path to the GROMACS .gro file
        water_resnames (tuple): Tuple of residue names considered as water molecules.
                               Default is ("SOL",) for SPC/E water model.

    Returns:
        tuple: (protein_count, water_count) - Number of protein and water residues

    Raises:
        FileNotFoundError: If the GRO file cannot be found
        ValueError: If the GRO file is malformed or too short

    Example:
        >>> prot, wat = count_residues_in_gro("complex.gro")
        >>> print(f"Protein: {prot}, Water: {wat}")
    """
    residue_ids = set()
    water_ids = set()

    try:
        with open(gro_path, "r") as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError(f"Invalid .gro file: {gro_path} - file too short")

        # Atom lines are from line 3 to N-2 (last two lines are box vectors)
        for line in lines[2:-1]:
            if len(line) < 20:
                continue  # skip malformed lines

            try:
                res_id = int(line[:5].strip())
                res_name = line[5:10].strip()

                if res_name in water_resnames:
                    water_ids.add(res_id)
                else:
                    residue_ids.add(res_id)
            except (ValueError, IndexError):
                continue  # skip malformed lines

        protein_count = len(residue_ids - water_ids)
        water_count = len(water_ids)

        return protein_count, water_count

    except FileNotFoundError:
        raise FileNotFoundError(f"GRO file not found: {gro_path}")
    except Exception as e:
        raise ValueError(f"Error parsing GRO file {gro_path}: {e}")


def tremd_temperature_ladder(
        Nw, Np, Tlow, Thigh, Pdes, WC=3, PC=1, Hff=0, Vs=0, Tol=0.001
):
    """
    Generate a temperature ladder for Temperature Replica Exchange Molecular Dynamics (T-REMD).

    This function implements the algorithm from Patriksson and van der Spoel (2008) to generate
    an optimized temperature ladder that maximizes exchange probability between adjacent replicas
    while maintaining the desired exchange probability Pdes.

    The algorithm considers:
    - System degrees of freedom based on constraints
    - Empirical coefficients for energy fluctuations
    - Numerical integration for exchange probability calculation
    - Iterative temperature adjustment for convergence

    Args:
        Nw (int): Number of water molecules in the system
        Np (int): Number of protein residues in the system
        Tlow (float): Minimum temperature (K) for the temperature ladder
        Thigh (float): Maximum temperature (K) for the temperature ladder
        Pdes (float): Desired exchange probability between adjacent replicas (0 < P < 1)
        WC (int): Water constraints (3 = all constraints, 2 = bonds only, 1 = none)
        PC (int): Protein constraints (1 = H atoms only, 2 = all bonds, 0 = none)
        Hff (int): Hydrogen force field switch (0 = standard, 1 = different model)
        Vs (int): Include volume correction (1 = yes, 0 = no)
        Tol (float): Tolerance for exchange probability convergence

    Returns:
        List[float]: List of temperatures forming the optimized T-REMD ladder

    Raises:
        ValueError: If input parameters are invalid or convergence fails

    Example:
        >>> temps = tremd_temperature_ladder(
        ...     Nw=1000, Np=200, Tlow=300, Thigh=400, Pdes=0.2
        ... )
        >>> print(f"Generated {len(temps)} temperature replicas")

    Note:
        This function automatically prints citation information for the original paper.
        The algorithm uses empirical coefficients that were fitted to molecular dynamics
        simulation data and may need adjustment for very different systems.
    """

    # Empirical coefficients from Patriksson and van der Spoel (2008)
    A0, A1 = -59.2194, 0.07594
    B0, B1 = -22.8396, 0.01347
    D0, D1 = 1.1677, 0.002976
    kB = 0.008314  # Boltzmann constant in kJ/mol/K
    maxiter = 100  # Maximum number of iterations for convergence

    # Estimate number of hydrogen atoms and virtual sites (VC) based on model
    if Hff == 0:
        Nh = round(Np * 0.5134)
        VC = round(1.91 * Nh) if Vs == 1 else 0
        Nprot = Np
    else:
        Npp = round(Np / 0.65957)
        Nh = round(Np * 0.22)
        VC = round(Np + 1.91 * Nh) if Vs == 1 else 0
        Nprot = Npp

    # Degrees of freedom corrections based on constraints
    NC = Nh if PC == 1 else Np if PC == 2 else 0
    Ndf = (9 - WC) * Nw + 3 * Np - NC - VC  # Total degrees of freedom
    FlexEner = 0.5 * kB * (NC + VC + WC * Nw)  # Internal flexibility energy

    # Probability evaluation function for exchange efficiency
    def myeval(m12, s12, CC, u):
        """Evaluate the exchange probability contribution at energy u."""
        arg = -CC * u - (u - m12) ** 2 / (2 * s12 ** 2)
        return np.exp(arg)

    # Numerical integration using midpoint method for exchange probability contribution
    def myintegral(m12, s12, CC):
        """Calculate the integral contribution to exchange probability."""
        umax = m12 + 5 * s12
        du = umax / 100
        u_vals = np.arange(0, umax, du)
        vals = [myeval(m12, s12, CC, u + du / 2) for u in u_vals]
        pi = np.pi
        return du * sum(vals) / (s12 * np.sqrt(2 * pi))

    # Initialize list of temperatures with the lowest value
    temps = [Tlow]

    # Iteratively compute the next temperature until reaching Thigh
    while temps[-1] < Thigh:
        T1 = temps[-1]  # Last accepted temperature
        T2 = T1 + 1 if T1 + 1 < Thigh else Thigh  # Initial guess for next temperature
        low, high = T1, Thigh
        iter_count = 0
        piter = 0
        forward = True  # Flag for adjusting search direction

        # Newton-like iteration to find T2 that yields desired exchange probability
        while abs(Pdes - piter) > Tol and iter_count < maxiter:
            iter_count += 1
            mu12 = (T2 - T1) * ((A1 * Nw) + (B1 * Nprot) - FlexEner)
            CC = (1 / kB) * ((1 / T1) - (1 / T2))
            var = Ndf * (D1 ** 2 * (T1 ** 2 + T2 ** 2) + 2 * D1 * D0 * (T1 + T2) + 2 * D0 ** 2)
            sig12 = np.sqrt(var)

            # Two components of the exchange probability
            I1 = 0.5 * math.erfc(mu12 / (sig12 * np.sqrt(2)))
            I2 = myintegral(mu12, sig12, CC)
            piter = I1 + I2

            # Adjust T2 up or down depending on current probability
            if piter > Pdes:
                if forward:
                    T2 += 1.0
                else:
                    low = T2
                    T2 = low + (high - low) / 2
                if T2 >= Thigh:
                    T2 = Thigh
            else:
                if forward:
                    forward = False
                    low = T2 - 1.0
                high = T2
                T2 = low + (high - low) / 2

        # Append rounded temperature to the list
        temps.append(round(T2, 2))

    print(
        "Please Cite: 'Alexandra Patriksson and David van der Spoel, A temperature predictor for parallel tempering "
        "\n"
        "simulations Phys. Chem. Chem. Phys., 10 pp. 2073-2077 (2008)'"
    )

    return temps


def print_help():
    """Print usage information and help for the T-REMD preparation tool."""
    print(
        """
T-REMD Temperature Ladder Generator
===================================

This tool generates optimized temperature ladders for Temperature Replica Exchange 
Molecular Dynamics (T-REMD) simulations based on the algorithm of Patriksson and 
van der Spoel (2008).

Usage:
  tremd_prep [complex.solvated.ions.gro]

Description:
  - Counts protein and water residues in the provided .gro file
  - Interactively prompts for temperature range and exchange probability
  - Generates and displays the optimized T-REMD temperature ladder
  - Saves the temperature ladder to 'TREMD_temp_ladder.txt'

Arguments:
  complex.solvated.ions.gro  Path to the GROMACS structure file

Interactive Prompts:
  - Initial temperature (K): Lowest temperature in the ladder
  - Final temperature (K): Highest temperature in the ladder  
  - Exchange probability: Desired probability between adjacent replicas (e.g., 0.2)

Output:
  - Console display of the temperature ladder
  - File: TREMD_temp_ladder.txt with replica temperatures

Reference:
  Alexandra Patriksson and David van der Spoel, "A temperature predictor for 
  parallel tempering simulations", Phys. Chem. Chem. Phys., 10 pp. 2073-2077 (2008)
  http://dx.doi.org/10.1039/b716554d

Example:
  $ python tremd_prep.py protein_solvated.gro
  Protein residues: 150
  Water residues: 2500
  Enter initial temperature (K): 300
  Enter final temperature (K): 400
  Enter desired exchange probability (e.g. 0.2): 0.2
  ...
"""
    )


def main():
    """
    Main function for T-REMD temperature ladder generation.

    This function provides the command-line interface for the T-REMD preparation tool.
    It handles argument parsing, file validation, user input, and output generation.

    Workflow:
    1. Parse command line arguments
    2. Validate input GRO file
    3. Count protein and water residues
    4. Collect user input for temperature parameters
    5. Generate temperature ladder
    6. Display results and save to file

    Raises:
        SystemExit: If arguments are invalid or file not found
    """
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    gro_file = sys.argv[1]
    if not os.path.exists(gro_file):
        print(f"[ERROR] File not found: {gro_file}")
        sys.exit(1)

    # Count residues in the system
    prot, wat = count_residues_in_gro(gro_file)
    print(f"Protein residues: {prot}")
    print(f"Water residues: {wat}")

    # Collect user input for temperature parameters
    try:
        Tlow = float(input("Enter initial temperature (K): "))
        Thigh = float(input("Enter final temperature (K): "))
        Pdes = float(input("Enter desired exchange probability (e.g. 0.2): "))
    except Exception as e:
        print(f"[ERROR] Invalid input: {e}")
        sys.exit(1)

    # Generate temperature ladder
    temps = tremd_temperature_ladder(Nw=wat, Np=prot, Tlow=Tlow, Thigh=Thigh, Pdes=Pdes)

    # Display results
    print("\nT-REMD Temperature Ladder:")
    for i, t in enumerate(temps):
        print(f"Replica {i + 1}: {t:.2f} K")

    # Write to file
    output_file = "TREMD_temp_ladder.txt"
    with open(output_file, "w") as f:
        for i, t in enumerate(temps):
            f.write(f"Replica {i + 1}: {t:.2f} K\n")
    print(f"\nTemperature ladder written to {output_file}")


if __name__ == "__main__":
    main()
