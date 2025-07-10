#!/bin/bash
#SBATCH --job-name=FEP_gmx_min
#SBATCH --output=slurm/fep_job_%j.out
#SBATCH --error=slurm/fep_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=0

module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# --- Parallel execution across lambda windows ---
i=0
for lambda_dir in lambda_*; do
    (
        cd "$lambda_dir" || exit

        # Extract lambda value from directory name (e.g., 0.00 from lambda_0.00)
        lambda_val=${lambda_dir#lambda_}
        lambda_tag=$(printf "hybrid_complex_%.2f" "$lambda_val")
        init="${lambda_tag}.solv.ions"
        mini_prefix="em_fep"
        nvt_prefix="nvt_fep"
        npt_prefix="npt_fep"

        if [ ! -f "${mini_prefix}.gro" ]; then
            echo "[${lambda_dir}] Running minimization..."
            gmx_mpi grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -maxwarn 20
            gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -deffnm ${mini_prefix}
        fi

        if [ ! -f "${nvt_prefix}.gro" ]; then
            echo "[${lambda_dir}] Running NVT..."
            gmx_mpi grompp -f ${nvt_prefix}.mdp -o ${nvt_prefix}.tpr -c ${mini_prefix}.gro -r ${init}.gro -p topol.top -maxwarn 20
            gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -deffnm ${nvt_prefix}
        fi

        if [ ! -f "${npt_prefix}.gro" ]; then
            echo "[${lambda_dir}] Running NPT..."
            gmx_mpi grompp -f ${npt_prefix}.mdp -o ${npt_prefix}.tpr -c ${nvt_prefix}.gro -r ${init}.gro -p topol.top -maxwarn 20
            gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -deffnm ${npt_prefix}
        fi

        echo "[${lambda_dir}] Done."

    ) &  # background job for each lambda window
    ((i++))

    # After every 21 jobs (1 per lambda), wait to stay within allocation
    if (( i % 21 == 0 )); then
        wait
    fi
done

wait
echo "All minimization and equilibration complete."
