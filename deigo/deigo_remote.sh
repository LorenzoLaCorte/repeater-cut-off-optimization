#!/bin/bash

# Description:
# This script is used to run the distillation_ml_gp.py script on the Deigo cluster.

REQUIREMENTS_FILE="requirements.txt"
SCRIPT_NAME="distillation_ml_gp.py"
REMOTE_DIR="experiments/cluster-sim-1"
PYTHON_VERSION="python/3.10.2"
SLURM_SCRIPT="ml_script.slurm"

cd $REMOTE_DIR

module load $PYTHON_VERSION
python3 -m pip install -r $REQUIREMENTS_FILE

rm $SLURM_SCRIPT
cat > $SLURM_SCRIPT <<- EOM
#!/bin/bash
#SBATCH -p compute
#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH -c 32
python3 $SCRIPT_NAME --gp_shots=100
EOM

sbatch $SLURM_SCRIPT