#!/bin/bash

# Description:
# This script is used to run the distillation_ml_gp.py script on the Deigo cluster.

REQUIREMENTS_FILE="requirements.txt"
SCRIPT_NAME="distillation_ml_gp.py"
REMOTE_HOST="deigo"
REMOTE_DIR="experiments/cluster-sim-1"
PYTHON_VERSION="python/3.10.2"
VENV_PATH="venv/bin/activate"
SLURM_SCRIPT="ml_script.slurm"

pip freeze > $REQUIREMENTS_FILE

scp -r $REQUIREMENTS_FILE $SCRIPT_NAME $REMOTE_HOST:$REMOTE_DIR

ssh $REMOTE_HOST
cd $REMOTE_DIR

module load $PYTHON_VERSION
python3 -m pip install $REQUIREMENTS_FILE

# TODO: add script content
cat > $SLURM_SCRIPT <<- EOM
#!/bin/bash
#SBATCH -p compute
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -c 8

python3 $SCRIPT_NAME
EOM
sbatch $SLURM_SCRIPT