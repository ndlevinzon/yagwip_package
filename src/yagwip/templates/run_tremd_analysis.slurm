#!/bin/bash
#SBATCH --job-name=remd_analysis
#SBATCH --output=remd_analysis_%j.out
#SBATCH --error=remd_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0

# A Decent Tutorial on TREMD with GROMACS: https://www.researchgate.net/publication/335453547_Tutorial_on_setting_up_the_Replica_Exchange_Molecular_Dynamics_REMD_simulation_using_alanine_dipeptide_as_a_toy_model #

# Load GROMACS module
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# ----------- SETUP -------------------
# Count directories that are named only with digits
num_dirs=$(find . -maxdepth 1 -type d -regex './[0-9]+' | wc -l)

# Print the result
echo "Found $num_dirs TREMD directories"


REPLICAS=$num_dirs
DEFFNM="remd"  # Trajectory/log base name

# ----------- STEP 0: DEMUX INDEX ---------------

echo "Generating replica index..."

mkdir -p remd_analysis_results

# Create a temporary directory for logs
LOG_TMP="remd_logs"
mkdir -p "$LOG_TMP"

# Copy remd.log from each replica directory and rename it numerically
for i in $(seq 0 $((REPLICAS - 1))); do
    cp $i/remd.log "$LOG_TMP/remd_$i.log"
done

# Concatenate all logs into a single file
cat "$LOG_TMP"/remd_*.log > "$LOG_TMP/REMD.log"

# De-multiplexing the logs, move them to analysis directory
demux.pl "$LOG_TMP/REMD.log" > remd_analysis_results/remd_replica_index.log
mv replica_index.xvg remd_analysis_results/
mv replica_temp.xvg remd_analysis_results/

# De-multiplexing the trajectories
echo "Demultiplexing trajectories..."

# Build list of input .xtc files
XTCLIST=()
for i in $(seq 0 $((REPLICAS - 1))); do
    XTCLIST+=("$i/remd.xtc")
done

# Make demuxed .xtc files for each replica and place them in their analysis directories
mpirun -np 40 trjcat -f "${XTCLIST[@]}" -demux remd_analysis_results/replica_index.xvg

# ----------- STEP 1: LOOP THROUGH REPLICAS ---------------

for TARGET_REPLICA in $(seq 0 $((REPLICAS - 1))); do
    echo "Processing replica $TARGET_REPLICA..."

    OUTDIR="remd_analysis_results/replica_${TARGET_REPLICA}"
    mkdir -p "$OUTDIR"

    # Move .xtc to the correct analysis directory
    mv ${TARGET_REPLICA}_trajout.xtc ${OUTDIR}

    DEMUX_TRAJ="$OUTDIR/${TARGET_REPLICA}_trajout.xtc"
    DEMUX_TPR="$OUTDIR/demuxed_${TARGET_REPLICA}.tpr"

    # 2. Copy matching .tpr
    cp ${TARGET_REPLICA}/${DEFFNM}.tpr "$DEMUX_TPR"

    # 3. RMSD
    echo 4 4 | mpirun -np 40 rms -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/rmsd.xvg"

    # 4. RMSF
    echo 4 | mpirun -np 40 rmsf -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/rmsf.xvg" -res

    # 5. Radius of Gyration
    echo 1 | mpirun -np 40 gyrate -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/gyrate.xvg"

    # 6. PCA
    echo 4 4 | mpirun -np 40 covar -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/eigenval.xvg" -v "$OUTDIR/eigenvec.trr"
    echo 4 4 | mpirun -np 40 anaeig -v "$OUTDIR/eigenvec.trr" -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -first 1 -last 2 -proj "$OUTDIR/proj.xvg"

    # 7. Clustering
    echo 4 4 | mpirun -np 40 cluster -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -method gromos -cutoff 0.2 \
        -o "$OUTDIR/clusters.xpm" -g "$OUTDIR/cluster.log" -sz "$OUTDIR/cluster_sizes.xvg"

    # 8. Free Energy Surface
    mpirun -np 40 sham -f "$OUTDIR/proj.xvg" -ls "$OUTDIR/free_energy.xpm" -bin 100

    # 9. Optional temperature extraction
    if [[ -f ${TARGET_REPLICA}/${DEFFNM}.edr ]]; then
        cp ${TARGET_REPLICA}/${DEFFNM}.edr "$OUTDIR/ener.edr"
        echo "Temperature" | mpirun -np 40 energy -f "$OUTDIR/ener.edr" -o "$OUTDIR/temp.xvg"
    fi

    echo "Replica ${TARGET_REPLICA} complete."
done

# Remove Cheackpoint Files
find . -type f -name '*#*' -exec rm -i {} \;

echo "All REMD replicas analyzed. Results in remd_analysis_results/"

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
