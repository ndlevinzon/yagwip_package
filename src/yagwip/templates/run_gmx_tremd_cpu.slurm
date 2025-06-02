#!/bin/bash
#SBATCH --job-name=gmx_remd
#SBATCH --output=slurm/remd_job_%j.out
#SBATCH --error=slurm/remd_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=36   # One per replica
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive 
#SBATCH --mem=0

# Load GROMACS module
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# Variables
init="step3_input"
mini_prefix="step4.0_minimization"
equi_prefix="step4.1_equilibration"
topol="topol.top"
index="index.ndx"

# Read temperatures from file formatted like: "Replica 1: 300.00 K"
temperature_file="TREMD_temp_ranges.txt"
mapfile -t temperatures < <(awk '{print $3}' "$temperature_file")
# ------------------------
# Step 1: Minimization
# ------------------------
if [ ! -f "${mini_prefix}.gro" ]; then
    echo "Starting energy minimization..."
    gmx grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -n index.ndx -maxwarn 1
    mpirun -np 32 gmx_mpi mdrun -v -deffnm ${mini_prefix}
else
    echo "Minimization output ${mini_prefix}.gro already exists. Skipping minimization."
fi

# ------------------------
# Step 2: Equilibration
# ------------------------
if [ ! -f "${equi_prefix}.gro" ]; then
    echo "Starting equilibration..."
    gmx grompp -f ${equi_prefix}.mdp -o ${equi_prefix}.tpr -c ${mini_prefix}.gro -r ${init}.gro -p topol.top -n index.ndx
    mpirun -np 32 gmx_mpi mdrun -v -deffnm ${equi_prefix}
else
    echo "Equilibration output ${equi_prefix}.gro already exists. Skipping equilibration."
fi

# ------------------------
# Step 3: Prepare REMD Directories
# ------------------------
echo "Preparing REMD directories..."
for i in ${!temperatures[@]}; do
    temp=${temperatures[$i]}
    dir=$i
    mkdir -p $dir
    cp ${topol} ${index} ${equi_prefix}.gro $dir/

    # Generate MDP for this temperature
    sed "s/XXX/${temp}/g" remd_template.mdp > $dir/remd.mdp
    
    # grompp for each replica
    pushd $dir
    gmx grompp -f remd.mdp -c ${equi_prefix}.gro -p ${topol} -n ${index} -o remd.tpr -maxwarn 1
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
    -ntomp 1 \
    -cpt 5 \
    -cpi remd.cpt

# ------------------------
# Step 5: Resubmit Job
# ------------------------
echo "Job ran for 55 minutes. Resubmitting self..."
sbatch run_gmx_tremd_beluga.slurm
