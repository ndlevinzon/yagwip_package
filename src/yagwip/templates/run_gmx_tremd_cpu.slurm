#!/bin/bash
#SBATCH --job-name=__BASE___gmx_remd
#SBATCH --output=slurm/remd_job_%j.out
#SBATCH --error=slurm/remd_job_%j.err
#SBATCH --nodes=8              # 8 Replicas = 8 Nodes
#SBATCH --ntasks-per-node=1    # 1 Node Per Replica
#SBATCH --cpus-per-task=40     # 40 CPUs Per Node
#SBATCH --time=01:00:00
#SBATCH --exclusive 
#SBATCH --mem=0

# Load GROMACS module
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# Variables
init="__INIT_GRO__"
mini_prefix="em"
nvt_prefix="nvt"
npt_prefix="npt"
prod_prefix="remd_template"
prod_step="step"

# Read temperatures from file formatted like: "Replica 1: 300.00 K"
temperature_file="TREMD_temp_ranges.txt"
mapfile -t temperatures < <(awk '{print $3}' "$temperature_file")

# ------------------------
# Step 1: Minimization
# ------------------------
if [ ! -f "${mini_prefix}.gro" ]; then
    echo "Starting energy minimization..."
    gmx grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -maxwarn 1
    mpirun -np 40 gmx_mpi mdrun -v -deffnm ${mini_prefix}
else
    echo "Minimization output ${mini_prefix}.gro already exists. Skipping minimization."
fi

# ------------------------
# Step 2: NVT
# ------------------------
if [ ! -f "${nvt_prefix}.gro" ]; then
    echo "Starting equilibration..."
    gmx grompp -f ${nvt_prefix}.mdp -o ${nvt_prefix}.tpr -c ${mini_prefix}.gro -r ${init}.gro -p topol.top
    mpirun -np 40 gmx_mpi mdrun -v -deffnm ${nvt_prefix}
else
    echo "Equilibration output ${nvt_prefix}.gro already exists. Skipping equilibration."
fi

# ------------------------
# Step 3: NPT
# ------------------------
if [ ! -f "${npt_prefix}.gro" ]; then
    echo "Starting equilibration..."
    gmx grompp -f ${npt_prefix}.mdp -o ${npt_prefix}.tpr -c ${nvt_prefix}.gro -r ${init}.gro -p topol.top
    mpirun -np 40 gmx_mpi mdrun -v -deffnm ${npt_prefix}
else
    echo "Equilibration output ${npt_prefix}.gro already exists. Skipping equilibration."
fi

# ------------------------
# Step 3: Prepare REMD Directories
# ------------------------
echo "Preparing REMD directories..."
for i in ${!temperatures[@]}; do
    temp=${temperatures[$i]}
    dir=$i
    mkdir -p $dir
    cp topol.top ${npt_prefix}.gro $dir/

    # Generate MDP for this temperature
    sed "s/XXX/${temp}/g" remd_template.mdp > $dir/remd.mdp

    # grompp for each replica
    pushd $dir
    gmx grompp -f remd.mdp -c ${npt_prefix}.gro -p topol.top -o remd.tpr -maxwarn 1
    popd

done

# ------------------------
# Step 4: Run T-REMD with checkpointing
# ------------------------
echo "Launching T-REMD with ${#temperatures[@]} replicas..."
timeout 3300 \
mpirun -np ${#temperatures[@]} gmx_mpi mdrun -multidir $(seq 0 $((${#temperatures[@]} - 1))) \
    -replex 1000 \
    -deffnm remd \
    -multi ${#temperatures[@]} \
    -ntomp 40 \
    -cpt 5 \
    -cpi remd.cpt

# ------------------------
# Step 5: Resubmit Job
# ------------------------
echo "Job ran for 55 minutes. Resubmitting self..."
sbatch run_gmx_tremd_cpu.slurm

#      ___          _____
#     /__/\        /  /::\
#     \  \:\      /  /:/\:\
#      \  \:\    /  /:/  \:\   ___     ___
#  _____\__\:\  /__/:/ \__\:| /__/\   /  /\
# /__/::::::::\ \  \:\ /  /:/ \  \:\ /  /:/
# \  \:\~~\~~\/  \  \:\  /:/   \  \:\  /:/
#  \  \:\  ~~~    \  \:\/:/     \  \:\/:/
#   \  \:\         \  \::/       \  \::/
#    \  \:\         \__\/         \__\/
#     \__\/