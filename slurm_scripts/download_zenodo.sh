#!/bin/bash
#SBATCH --job-name=zenodo_7846094
#SBATCH --output=zenodo_7846094.log
#SBATCH --time=72:00:00         # Up to 72 hours
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1

# Target directory
OUTDIR="/share/mihaela-larisa.clement/soeampc-data"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Starting downloads into $OUTDIR"
echo "-----------------------------------------------------"

# Use wget with resume (-c), quiet mode (-q), and show progress bar (--show-progress)
wget -c --show-progress https://zenodo.org/api/records/7846094/files/chain_mass_3_N_120000_test.tar.lzip/content -O chain_mass_3_N_120000_test.tar.lzip
wget -c --show-progress https://zenodo.org/api/records/7846094/files/chain_mass_3_N_19080000.tar.lzip/content -O chain_mass_3_N_19080000.tar.lzip
wget -c --show-progress https://zenodo.org/api/records/7846094/files/pretrained_models.zip/content -O pretrained_models.zip
wget -c --show-progress https://zenodo.org/api/records/7846094/files/quadcopter_N_120000_test.tar.lzip/content -O quadcopter_N_120000_test.tar.lzip
wget -c --show-progress https://zenodo.org/api/records/7846094/files/quadcopter_N_9600000.tar.lzip/content -O quadcopter_N_9600000.tar.lzip
wget -c --show-progress https://zenodo.org/api/records/7846094/files/stirtank_N_240000_test.tar.lzip/content -O stirtank_N_240000_test.tar.lzip
wget -c --show-progress https://zenodo.org/api/records/7846094/files/stirtank_N_960000.tar.lzip/content -O stirtank_N_960000.tar.lzip

echo "-----------------------------------------------------"
echo "All downloads completed (or resumed successfully)."

# Optionally verify integrity with md5sum:
echo "Verifying MD5 checksums..."
md5sum -c <<EOF
6662384640fdb5b691cf385a71452ea3  chain_mass_3_N_120000_test.tar.lzip
2e74a362bb265bc29f42be27b8fde34e  chain_mass_3_N_19080000.tar.lzip
cdb8a3d52373f4e467a488da5936e1be  pretrained_models.zip
b81bb4c9cf9fd2ca313284379424ce64  quadcopter_N_120000_test.tar.lzip
a4c126679cccb677024d237f5360c2c6  quadcopter_N_9600000.tar.lzip
6f8be2c44842800208a1f34af42e705f  stirtank_N_240000_test.tar.lzip
2f75d67b37e9e8fa836bce2535c6b404  stirtank_N_960000.tar.lzip
EOF

