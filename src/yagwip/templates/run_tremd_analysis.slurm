#!/bin/bash

# A Decent Tutorial on TREMD with GROMACS: https://www.researchgate.net/publication/335453547_Tutorial_on_setting_up_the_Replica_Exchange_Molecular_Dynamics_REMD_simulation_using_alanine_dipeptide_as_a_toy_model #

# ------------------------
# Load GROMACS module
# ------------------------
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# ------------------------
# SETUP
# ------------------------
# Count directories that are named only with digits
num_dirs=$(find . -maxdepth 1 -type d -regex './[0-9]+' | wc -l)

# Print the result
echo "Found $num_dirs TREMD directories"


REPLICAS=$num_dirs
DEFFNM="remd"  # Trajectory/log base name

# ------------------------
# GENERATE DEMUX INDEX
# ------------------------

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
gmx trjcat -f "${XTCLIST[@]}" -demux remd_analysis_results/replica_index.xvg

# ------------------------
# LOOP THROUGH REPLICAS
# ------------------------

for TARGET_REPLICA in $(seq 0 $((REPLICAS - 1))); do
    echo "Processing replica $TARGET_REPLICA..."

    OUTDIR="remd_analysis_results/replica_${TARGET_REPLICA}"
    mkdir -p "$OUTDIR"

    # 1. Move .xtc to the correct analysis directory
    mv ${TARGET_REPLICA}_trajout.xtc ${OUTDIR}

    DEMUX_TRAJ="$OUTDIR/${TARGET_REPLICA}_trajout.xtc"
    DEMUX_TPR="$OUTDIR/demuxed_${TARGET_REPLICA}.tpr"

    # 2. Copy matching .tpr
    cp ${TARGET_REPLICA}/${DEFFNM}.tpr "$DEMUX_TPR"

    # 3. RMSD
    echo 4 4 | gmx rms -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/rmsd.xvg" -res

    # 4. RMSF
    echo 4 | gmx rmsf -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/rmsf.xvg" -res

    # 5. PCA
    echo 0 0 | gmx trjconv -s "$DEMUX_TPR" -f "$DEMUX_TRAJ" -o "$OUTDIR/traj_centered.xtc" -center -pbc mol
    echo 0 0 | gmx trjconv -s "$DEMUX_TPR" -f "$OUTDIR/traj_centered.xtc" -o "$OUTDIR/traj_centered_rot_trans.xtc" -ur compact -fit rot+trans
    echo 4 4 | gmx covar -s "$DEMUX_TPR" -f "$OUTDIR/traj_centered_rot_trans.xtc" -o "$OUTDIR/eigenval.xvg" -v "$OUTDIR/eigenvec.trr"
    echo 4 4 | gmx anaeig -v "$OUTDIR/eigenvec.trr" -s "$DEMUX_TPR" -f "$OUTDIR/traj_centered_rot_trans.xtc" -proj "$OUTDIR/proj.xvg"

    # 6. Temperature extraction
    if [[ -f ${TARGET_REPLICA}/${DEFFNM}.edr ]]; then
        cp ${TARGET_REPLICA}/${DEFFNM}.edr "$OUTDIR/ener.edr"
        echo "Temperature" | gmx energy -f "$OUTDIR/ener.edr" -o "$OUTDIR/temp.xvg"
    fi

    echo "Replica ${TARGET_REPLICA} complete."
done

# ------------------------
# COMBINE ENERGY FILES
# ------------------------
# Build list of input .edr files
EDRLIST=()
for i in $(seq 0 $((REPLICAS - 1))); do
    EDRLIST+=("$i/remd.edr")
done
gmx eneconv -f "${EDRLIST[@]}" -o remd_analysis_results/combined.edr
echo "Potential" | gmx energy -f remd_analysis_results/combined.edr -o remd_analysis_results/kbT_scalar.xvg


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
