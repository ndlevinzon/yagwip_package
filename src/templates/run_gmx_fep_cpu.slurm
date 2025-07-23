#!/bin/bash
#SBATCH --job-name=FEP_production
#SBATCH --output=slurm/md_job_%j.out
#SBATCH --error=slurm/md_job_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=85
#SBATCH --cpus-per-task=3
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --mem=0

module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

i=0

for system in ligand_only protein_complex; do
  for direction in A_to_B B_to_A; do
    for lambda_dir in ${system}/${direction}/lambda_*; do
      (
        cd "$lambda_dir" || exit

        lambda_val=${lambda_dir##*/lambda_}
        lambda_tag=$(printf "hybrid_%.2f" "$lambda_val")
        npt_prefix="npt_fep"
        prod_prefix="production_fep"

        # Prepare production TPR
        if [ ! -f "${prod_prefix}.tpr" ]; then
          echo "[${lambda_dir}] Generating production TPR..."
          gmx_mpi grompp -f ${prod_prefix}.mdp -c ${npt_prefix}.gro -p topol.top -o ${prod_prefix}.tpr -maxwarn 20
        fi

        # Run production
        echo "[${lambda_dir}] Running production MD..."
        gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -deffnm ${prod_prefix} -cpt 60 -maxh 2.92

        echo "[${lambda_dir}] Production complete."
      ) &

      ((i++))

      # Wait after every 85 concurrent jobs
      if (( i % 85 == 0 )); then
        wait
      fi
    done
  done
done

wait
echo "All production FEP simulations complete."
