#!/bin/bash

# Note: to be called from the root directory of the project
# Description:
# This script is used to run the distillation_ml_gp.py script on the Deigo cluster.

REQUIREMENTS_FILE="requirements.txt"
SCRIPT_NAME="distillation_ml_gp.py"
BASH_SCRIPT_NAME="deigo/deigo_remote.sh"
REMOTE_HOST="deigo"
REMOTE_DIR="experiments/cluster-sim-1"

pip freeze > $REQUIREMENTS_FILE
scp -r $REQUIREMENTS_FILE $SCRIPT_NAME $BASH_SCRIPT_NAME $REMOTE_HOST:$REMOTE_DIR