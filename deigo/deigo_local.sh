#!/bin/bash

# Note: to be called from the root directory of the project
# Description:
# This script is used to move the required scripts on the Deigo cluster.

CONFIG_FILE="config.json"
REQUIREMENTS_FILE="requirements.txt"
SETUP_SCRIPT_NAME="deigo/deigo_remote_setup.sh"
RUNNER_SCRIPT_NAME="gp_symmetric_runner.sh"

RUN_SCRIPT_NAME="deigo/deigo_remote_run.sh"

REMOTE_HOST="deigo"
REMOTE_DIR="experiments/cluster-sim-1"

pip freeze > $REQUIREMENTS_FILE
scp -r $CONFIG_FILE $REQUIREMENTS_FILE $RUNNER_SCRIPT_NAME $SETUP_SCRIPT_NAME $RUN_SCRIPT_NAME $REMOTE_HOST:$REMOTE_DIR
scp -r *.py $REMOTE_HOST:$REMOTE_DIR