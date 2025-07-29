#!/bin/bash
#SBATCH --job-name=__BASE___gmx_tremd
#SBATCH --output=slurm/remd_job_%j.out
#SBATCH --error=slurm/remd_job_%j.err
#SBATCH --nodes=8              # Nodes
#SBATCH --ntasks-per-node=4    # Replicas per Node
#SBATCH --cpus-per-task=16     # CPUs Per Replicas
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=0

export OMP_NUM_THREADS=16

# ------------------------
# Load GROMACS module
# ------------------------
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# ------------------------
# VARIABLES
# ------------------------
npt_prefix="npt"
prod_prefix="production_remd"
prod_step="step"

# ------------------------
# READ TEMPERATURES FROM TEMPERATURE LADDER FILE
# ------------------------
temperature_file="TREMD_temp_ranges.txt"
mapfile -t temperatures < <(awk '{print $3}' "$temperature_file")

# ------------------------
# PREPARE T-REMD DIRECTORIES
# ------------------------
echo "Preparing T-REMD directories..."
for i in ${!temperatures[@]}; do
    temp=${temperatures[$i]}
    dir=$i
    mkdir -p $dir
    cp topol.top ${npt_prefix}.gro $dir/

    # Generate MDP for this temperature
    sed "s/XXX/${temp}/g" ${prod_prefix}.mdp > $dir/remd.mdp

    # grompp for each replica
    pushd $dir
    srun gmx_mpi grompp -f remd.mdp -c ${npt_prefix}.gro -p topol.top -o remd.tpr -maxwarn 20
    popd

done

# ------------------------
# RUN T-REMD WITH CHECKPOINTING
# ------------------------
echo "Launching T-REMD with ${#temperatures[@]} replicas..."
srun gmx_mpi mdrun -multidir $(seq 0 $((${#temperatures[@]} - 1))) \
  -replex 500 \
  -deffnm remd \
  -ntomp ${SLURM_CPUS_PER_TASK} \
  -cpt 5 \
  -cpi remd.cpt \
  -maxh 0.92

echo "Job ran for 55 minutes. Resubmitting self..."
sbatch run_gmx_tremd_cpu.slurm

#--------------------------------------
#__  _____   _______       __________
#\ \/ /   | / ____/ |     / /  _/ __ \
# \  / /| |/ / __ | | /| / // // /_/ /
# / / ___ / /_/ / | |/ |/ // // ____/
#/_/_/  |_\____/  |__/|__/___/_/
#--------------------------------------