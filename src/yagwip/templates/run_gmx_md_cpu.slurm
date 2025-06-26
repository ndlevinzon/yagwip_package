#!/bin/bash
#SBATCH --job-name=gmx_md
#SBATCH --output=slurm/md_job_%j.out
#SBATCH --error=slurm/md_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=40   # Max on Beluga
#SBATCH --cpus-per-task=1
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
prod_prefix="prod"
prod_step="step"

# ------------------------
# Step 1: Minimization
# ------------------------
if [ ! -f "${mini_prefix}.gro" ]; then
    echo "Starting energy minimization..."
    gmx grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -n index.ndx -maxwarn 1
    mpirun -np 40 gmx_mpi mdrun -v -deffnm ${mini_prefix}
else
    echo "Minimization output ${mini_prefix}.gro already exists. Skipping minimization."
fi

# ------------------------
# Step 2: NVT
# ------------------------
if [ ! -f "${nvt_prefix}.gro" ]; then
    echo "Starting equilibration..."
    gmx grompp -f ${nvt_prefix}.mdp -o ${nvt_prefix}.tpr -c ${mini_prefix}.gro -r ${init}.gro -p topol.top -n index.ndx
    mpirun -np 40 gmx_mpi mdrun -v -deffnm ${nvt_prefix}
else
    echo "Equilibration output ${nvt_prefix}.gro already exists. Skipping equilibration."
fi

# ------------------------
# Step 3: NPT
# ------------------------
if [ ! -f "${npt_prefix}.gro" ]; then
    echo "Starting equilibration..."
    gmx grompp -f ${npt_prefix}.mdp -o ${npt_prefix}.tpr -c ${nvt_prefix}.gro -r ${init}.gro -p topol.top -n index.ndx
    mpirun -np 40 gmx_mpi mdrun -v -deffnm ${npt_prefix}
else
    echo "Equilibration output ${npt_prefix}.gro already exists. Skipping equilibration."
fi

# ------------------------
# Step 4: Production (10 segments)
# ------------------------
echo "Starting production MD..."

cnt=1
cntmax=10

start_time=$(date +%s)
max_runtime=$((55 * 60))  # 55 minutes in seconds

while [ $cnt -le $cntmax ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $max_runtime ]; then
        echo "Reached 55-minute runtime limit. Resubmitting and exiting..."
        sbatch "$0"
        exit 0
    fi

    pcnt=$((cnt - 1))
    istep="${prod_step}_${cnt}"

    if [ -f "${istep}.gro" ]; then
        echo "${istep}.gro exists. Skipping this step."
        cnt=$((cnt + 1))
        continue
    fi

    if [ $cnt -eq 1 ]; then
        pstep=$npt_prefix
        mpirun -np 40 gmx_mpi grompp -f "${prod_prefix}.mdp" -o "${istep}.tpr" -c "${pstep}.gro" -p topol.top -n index.ndx
    else
        pstep="${prod_step}_${pcnt}"
        mpirun -np 40 gmx_mpi grompp -f "${prod_prefix}.mdp" -o "${istep}.tpr" -c "${pstep}.gro" -t "${pstep}.cpt" -p topol.top -n index.ndx
    fi

    echo "Running production chunk: ${istep}"
    mpirun -np 40 gmx_mpi mdrun -v -deffnm "${istep}" -cpi "${istep}.cpt" &

    mdrun_pid=$!
    while kill -0 $mdrun_pid 2> /dev/null; do
        sleep 60  # check every 60 seconds
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [ $elapsed -ge $max_runtime ]; then
            echo "Time exceeded during mdrun. Killing job and resubmitting..."
            kill $mdrun_pid
            wait $mdrun_pid 2>/dev/null
            sbatch "$0"
            exit 0
        fi
    done

    echo "${istep} complete."
    cnt=$((cnt + 1))
done

#--------------------------------------
#__  _____   _______       __________
#\ \/ /   | / ____/ |     / /  _/ __ \
# \  / /| |/ / __ | | /| / // // /_/ /
# / / ___ / /_/ / | |/ |/ // // ____/
#/_/_/  |_\____/  |__/|__/___/_/
#--------------------------------------