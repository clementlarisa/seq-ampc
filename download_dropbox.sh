#!/bin/bash
#SBATCH --job-name=dropbox_tfrecord
#SBATCH --output=dropbox_tfrecord.log
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1

OUTDIR="/share/mihaela-larisa.clement/mit-tfrecord"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Starting Dropbox download into $OUTDIR"
echo "-----------------------------------------------------"

# Dropbox folder link (force download as zip by using dl=1)
DROPBOX_LINK="https://www.dropbox.com/scl/fi/glca7lqczsjmymgn2aqw0/tf_record_seq.zip?rlkey=96vyy53c7rawtmc0ojnh1ggmw&e=1&dl=1"

# Output filename
OUTFILE="tfrecord_dropbox_folder.zip"

# Download with resume support and progress bar
wget -c --show-progress "$DROPBOX_LINK" -O "$OUTFILE"

echo "-----------------------------------------------------"
echo "Download completed: $OUTFILE"

# Optional: unzip
echo "Unzipping..."
unzip -o "$OUTFILE"

echo "Done."
