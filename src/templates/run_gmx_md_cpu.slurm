#!/bin/bash
#SBATCH --job-name=__BASE___gmx_md
#SBATCH --output=slurm/md_job_%j.out
#SBATCH --error=slurm/md_job_%j.err
#SBATCH --nodes=1              # Number of Nodes
#SBATCH --ntasks-per-node=8    # Replicas per Node
#SBATCH --cpus-per-task=8      # CPUs Per Replicas
#SBATCH --time=3:00:00        # Time limit (12 hour)
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
prod_prefix="production"
prod_step="step"

# ------------------------
# Production (10 segments)
# ------------------------
echo "Starting production MD..."

srun gmx_mpi mdrun -deffnm ${prod_prefix} -cpi prod.cpt -maxh 2.92 -nsteps -1

sbatch run_gmx_md_cpu.slurm


