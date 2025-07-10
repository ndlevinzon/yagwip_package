#!/bin/bash
#SBATCH --job-name=FEP_gmx_min
#SBATCH --output=slurm/fep_job_%j.out
#SBATCH --error=slurm/fep_job_%j.err
#SBATCH --nodes=1              # Number of Nodes
#SBATCH --ntasks-per-node=21    # Replicas per Node
#SBATCH --cpus-per-task=3     # CPUs Per Replicas
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
mini_prefix="em_fep"
nvt_prefix="nvt_fep"
npt_prefix="npt_fep"

# ------------------------
# ADJUST TOPOL FILE TO CURRENT WORKING DIRECTORY
# ------------------------
# Step 1: Capture the current working directory in a variable
CUR_DIR="$(pwd)"

# Step 2: Make a backup copy of topol.top in case anything goes wrong
cp topol.top topol.top.bak

# Step 3: Use awk to scan through the lines of the backup file and modify #include paths
awk -v cwd="$CUR_DIR" '
function dirname(path,    n) {
    n = split(path, parts, "/")
    parts[n] = ""             # remove last element
    return parts[1] (n > 2 ? "/" parts[2] : "") (n > 3 ? "/" parts[3] : "") (n > 4 ? "/" parts[4] : "")
}

{
    if ($1 == "#include") {
        path = $2
        gsub(/"/, "", path)

        # Case 1: Relative path starting with "./"
        if (path ~ /^\.\/.*/) {
            new_path = "\"" cwd "/" substr(path, 3) "\""
            print "#include " new_path
        }

        # Case 2: Relative path starting with "../"
        else if (path ~ /^\.\.\//) {
            parent_dir = cwd
            sub(/\/$/, "", parent_dir)                 # Remove trailing slash if present
            n = split(parent_dir, parts, "/")
            if (n > 1) {
                up_dir = parts[1]
                for (i = 2; i < n; i++) {
                    up_dir = up_dir "/" parts[i]
                }
            } else {
                up_dir = "/"    # Already at root
            }
            rel_path = substr(path, 4)                 # Remove "../"
            new_path = "\"" up_dir "/" rel_path "\""
            print "#include " new_path
        }

        # Case 3: Absolute path that does not begin with cwd
        else if (path ~ /^\// && index(path, cwd) != 1) {
            file = substr(path, length(path) - index(reverse(path), "/") + 2)
            new_path = "\"" cwd "/" file "\""
            print "#include " new_path
        }

        # Case 4: Path is already absolute and within cwd
        else {
            print $0
        }
    } else {
        print $0
    }
}

function reverse(s,    i, r) {
    r = ""
    for (i = length(s); i > 0; i--) {
        r = r substr(s, i, 1)
    }
    return r
}
' topol.top.bak > topol.top
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

echo "Minimization and Equilibration Done. Submitting TREMD..."
sbatch run_gmx_tremd_cpu.slurm

#--------------------------------------
#__  _____   _______       __________
#\ \/ /   | / ____/ |     / /  _/ __ \
# \  / /| |/ / __ | | /| / // // /_/ /
# / / ___ / /_/ / | |/ |/ // // ____/
#/_/_/  |_\____/  |__/|__/___/_/
#--------------------------------------