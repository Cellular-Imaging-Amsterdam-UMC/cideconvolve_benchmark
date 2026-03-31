#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.24gb
#SBATCH --mem 52gb
#SBATCH --time 8:00:00
#SBATCH --output=cideconvolve_%A.log

# ---------------------------------------------------------------------------
# CIDeconvolve SLURM job script
#
# Usage:
#   sbatch cideconvolve.slurm \
#       --infolder /path/to/input \
#       --outfolder /path/to/output \
#       [-- <extra wrapper.py flags>]
#
# Example:
#   sbatch cideconvolve.slurm \
#       --infolder /data/myimages \
#       --outfolder /data/results \
#       -- --method sdeconv_rl --iterations 40 --benchmark True \
#          --bench_methods all --na auto --refractive_index auto \
#          --sample_ri auto --microscope_type auto \
#          --emission_wl auto --excitation_wl auto
#
# Tip: The launcher.py "Command preview" shows the full docker command.
#      Copy everything after the image name and paste it as extra flags
#      after the -- separator above.
# ---------------------------------------------------------------------------

# --- Parse required folder arguments --------------------------------------
INFOLDER=""
OUTFOLDER=""
GTFOLDER=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --infolder)  INFOLDER="$2";  shift 2 ;;
        --outfolder) OUTFOLDER="$2"; shift 2 ;;
        --gtfolder)  GTFOLDER="$2";  shift 2 ;;
        --)          shift; EXTRA_ARGS="$*"; break ;;
        *)           EXTRA_ARGS="$*"; break ;;
    esac
done

if [ -z "$INFOLDER" ] || [ -z "$OUTFOLDER" ]; then
    echo "Error: --infolder and --outfolder are required."
    echo "Usage: sbatch cideconvolve.slurm --infolder /path/in --outfolder /path/out [-- <extra flags>]"
    exit 1
fi

# Default gtfolder to a temp dir if not specified
if [ -z "$GTFOLDER" ]; then
    GTFOLDER=$(mktemp -d)
    echo "No --gtfolder specified, using temp: $GTFOLDER"
fi

# --- Singularity image ----------------------------------------------------
# Pull once from Docker Hub; reuse the cached .sif on subsequent runs.
IMAGE_NAME="cellularimagingcf/w_cideconvolve"
SIF_DIR="${HOME}/.singularity/cideconvolve"
SIF_PATH="${SIF_DIR}/w_cideconvolve.sif"

mkdir -p "$SIF_DIR"
if [ ! -f "$SIF_PATH" ]; then
    echo "Pulling Singularity image from docker://${IMAGE_NAME} ..."
    singularity pull "$SIF_PATH" "docker://${IMAGE_NAME}"
fi

# --- Run -------------------------------------------------------------------
echo "============================================================"
echo "CIDeconvolve SLURM job"
echo "============================================================"
echo "  Input  : $INFOLDER"
echo "  Output : $OUTFOLDER"
echo "  GT     : $GTFOLDER"
echo "  Extra  : $EXTRA_ARGS"
echo "  Image  : $SIF_PATH"
echo "============================================================"

singularity exec --nv \
    --bind "${INFOLDER}:/data/in" \
    --bind "${OUTFOLDER}:/data/out" \
    --bind "${GTFOLDER}:/data/gt" \
    "$SIF_PATH" \
    python /app/wrapper.py \
        --infolder /data/in \
        --outfolder /data/out \
        --gtfolder /data/gt \
        --local \
        $EXTRA_ARGS
