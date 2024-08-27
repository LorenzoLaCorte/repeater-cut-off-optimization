#!/bin/bash

# Description:
# This script is used to run a script on the Deigo cluster.

SLURM_SCRIPT="script.slurm"

rm $SLURM_SCRIPT
cat > $SLURM_SCRIPT <<- EOM
#!/bin/bash
#SBATCH -p compute
#SBATCH -t 48:00:00
#SBATCH --mem=256G
#SBATCH -c 32
./distillation_results.sh
EOM

sbatch $SLURM_SCRIPT