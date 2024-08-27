#!/bin/bash

# Description:
# This script is used to setup the environment on the Deigo cluster.

REQUIREMENTS_FILE="requirements.txt"
PYTHON_VERSION="python/3.10.2"

module load $PYTHON_VERSION
python3.10 -m pip install -r $REQUIREMENTS_FILE