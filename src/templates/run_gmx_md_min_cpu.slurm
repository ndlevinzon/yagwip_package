#!/bin/bash
#SBATCH --job-name=__BASE___gmx_min
#SBATCH --output=slurm/md_job_%j.out
#SBATCH --error=slurm/md_job_%j.err
#SBATCH --nodes=1              # Number of Nodes
#SBATCH --ntasks-per-node=1    # Replicas per Node
#SBATCH --cpus-per-task=64     # CPUs Per Replicas
#SBATCH --time=01:00:00        # Time limit (1 hour)
#SBATCH --exclusive            # Exclusive node allocation
#SBATCH --mem=0                # Use all available memory

# ------------------------
# Load GROMACS module
# ------------------------
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# ------------------------
# VARIABLES
# ------------------------
init="__INIT__.solv.ions"
mini_prefix="em"
nvt_prefix="nvt"
npt_prefix="npt"

# ------------------------
# ADJUST TOPOL FILE TO CURRENT WORKING DIRECTORY
# ------------------------
# Step 1: Capture the current working directory in a variable
CUR_DIR="$(pwd)"

# Step 2: Make a backup copy of topol.top in case anything goes wrong
cp topol.top topol.top.bak

# Step 3: Use awk to scan through the lines of the backup file and modify #include paths
awk -v cwd="$CUR_DIR" '                                   # Pass the current directory to awk as variable "cwd"
{
    if ($1 == "#include") {                               # Process only lines that begin with "#include"
        path = $2                                         # Extract the path portion (e.g., "./amber14sb.ff/forcefield.itp")
        gsub(/"/, "", path)                               # Remove any quotation marks around the path

        # Case 1: Handle relative paths that start with "./"
        if (path ~ /^\.\/.*/) {
            new_path = "\"" cwd "/" substr(path, 3) "\""  # Remove "./" from the beginning and prepend the current directory
            print "#include " new_path                    # Print the modified line
        }

        # Case 2: Handle absolute paths that do NOT begin with the current directory
        else if (path ~ /^\// && index(path, cwd) != 1) {
            # Extract just the file path component (e.g., "amber14sb.ff/forcefield.itp")
            file = substr(path, length(path) - index(reverse(path), "/") + 2)

            # Construct new absolute path using the current directory + file
            new_path = "\"" cwd "/" file "\""
            print "#include " new_path
        }

        # Case 3: Path is already absolute and within current directory; leave unchanged
        else {
            print $0
        }
    } else {
        # For all other lines (not #include), print them unchanged
        print $0
    }
}

# Helper function: reverse(string)
# Used to locate the last "/" in the path string
function reverse(s,    i, r) {
    r = ""
    for (i = length(s); i > 0; i--) {
        r = r substr(s, i, 1)
    }
    return r
}
' topol.top.bak > topol.top    # Output the modified content into topol.top, replacing the original

# ------------------------
# MINIMIZATION
# ------------------------
if [ ! -f "${mini_prefix}.gro" ]; then
    echo "Starting energy minimization..."
    srun gmx_mpi grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -maxwarn 20
    srun gmx_mpi mdrun -v -ntomp ${SLURM_CPUS_PER_TASK} -deffnm ${mini_prefix}
else
    echo "Minimization output ${mini_prefix}.gro already exists. Skipping minimization."
fi

# ------------------------
# NVT
# ------------------------
if [ ! -f "${nvt_prefix}.gro" ]; then
    echo "Starting equilibration..."
    srun gmx_mpi grompp -f ${nvt_prefix}.mdp -o ${nvt_prefix}.tpr -c ${mini_prefix}.gro -r ${init}.gro -p topol.top -maxwarn 20
    srun gmx_mpi mdrun -v -ntomp ${SLURM_CPUS_PER_TASK} -deffnm ${nvt_prefix}
else
    echo "Equilibration output ${nvt_prefix}.gro already exists. Skipping equilibration."
fi

# ------------------------
# NPT
# ------------------------
if [ ! -f "${npt_prefix}.gro" ]; then
    echo "Starting equilibration..."
    srun gmx_mpi grompp -f ${npt_prefix}.mdp -o ${npt_prefix}.tpr -c ${nvt_prefix}.gro -r ${init}.gro -p topol.top -maxwarn 20
    srun gmx_mpi mdrun -v -ntomp ${SLURM_CPUS_PER_TASK} -deffnm ${npt_prefix}
else
    echo "Equilibration output ${npt_prefix}.gro already exists. Skipping equilibration."
fi

echo "Minimization and Equilibration Done. Submitting MD..."
sbatch run_gmx_md_cpu.slurm

#--------------------------------------
#__  _____   _______       __________
#\ \/ /   | / ____/ |     / /  _/ __ \
# \  / /| |/ / __ | | /| / // // /_/ /
# / / ___ / /_/ / | |/ |/ // // ____/
#/_/_/  |_\____/  |__/|__/___/_/
#--------------------------------------