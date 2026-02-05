#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=150:00:00
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/slurm-%j.out
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=YOUR_QOS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_PATH="${STORAGE_PATH:-$REPO_ROOT/data}"

# Activate your environment before running, e.g.:
# source /path/to/conda.sh && conda activate swirl

cd "$REPO_ROOT/preprocess"

# Optional: add custom frame extraction scripts here if available.

python extract_frames_VidGen.py -N 10 -k 1 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 2 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 3 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 4 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 5 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 6 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 7 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 8 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 9 --storage_path "$STORAGE_PATH"
python extract_frames_VidGen.py -N 10 -k 10 --storage_path "$STORAGE_PATH"
