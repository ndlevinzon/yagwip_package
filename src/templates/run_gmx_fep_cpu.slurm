#!/bin/bash
#SBATCH --job-name=FEP_gmx_prod
#SBATCH --output=slurm/fep_prod_%j.out
#SBATCH --error=slurm/fep_prod_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=21              # 21 lambda windows
#SBATCH --cpus-per-task=3        # 3 threads per window
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0

module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# ------------------------
# RUN PRODUCTION FOR EACH LAMBDA WINDOW
# ------------------------

i=0
for lambda_dir in lambda_*; do
    (
        cd "$lambda_dir" || exit

        # Extract lambda value and construct hybrid init name
        lambda_val=${lambda_dir#lambda_}
        lambda_tag=$(printf "hybrid_complex_%.2f" "$lambda_val")
        init="${lambda_tag}.solv.ions"
        prod_prefix="production_fep"

        # Only rerun grompp if .tpr doesn't exist
        if [ ! -f ${prod_prefix}.tpr ]; then
            echo "[${lambda_dir}] Preparing TPR..."
            gmx_mpi grompp -f production_fep.mdp -c npt_fep.gro -r ${init}.gro -p topol.top -o ${prod_prefix}.tpr -maxwarn 20
        fi

        echo "[${lambda_dir}] Starting production run..."
        gmx_mpi mdrun -deffnm ${prod_prefix} -ntomp $SLURM_CPUS_PER_TASK

        echo "[${lambda_dir}] Done."

    ) &  # Launch each in the background
    ((i++))

    # Throttle after 21 jobs (1 per lambda)
    if (( i % 21 == 0 )); then
        wait
    fi
done

wait
echo "FEP production completed for all lambda windows."
