#!/bin/bash
#SBATCH --job-name=__BASE___gmx_md
#SBATCH --output=slurm/md_job_%j.out
#SBATCH --error=slurm/md_job_%j.err
#SBATCH --account=rrg-mailhoto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --exclusive
#SBATCH --mem=0

# ------------------------
# Load GROMACS module
# ------------------------
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# ------------------------
# VARIABLES
# ------------------------
prod_prefix="production"
control_file="control.txt"

# ------------------------
# Initialize control.txt if it doesn't exist
# ------------------------
if [ ! -f "$control_file" ]; then
    echo "go" > "$control_file"
fi

# ------------------------
# Run production MD
# ------------------------
echo "Starting production MD..."
gmx mdrun -v -deffnm ${prod_prefix} -cpi ${prod_prefix}_prev -maxh 2.92 -nsteps -1

# ------------------------
# Resubmit if control.txt says "go"
# ------------------------
if grep -Fxq "go" "$control_file"; then
    echo "Resubmitting job..."
    sbatch "$0"
else
    echo "Job stopped by control.txt"
fi
