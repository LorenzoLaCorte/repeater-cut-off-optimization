#!/bin/bash

# Update package list
sudo apt update

# Install Python and pip
sudo apt install -y python3 python3-pip

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
pip3 install -r requirements.txt