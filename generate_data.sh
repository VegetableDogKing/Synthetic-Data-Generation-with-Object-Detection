#!/bin/bash

# This is the path where Isaac Sim is installed which contains the python.sh script
ISAAC_SIM_PATH="/home/vegetabledogking/.local/share/ov/pkg/isaac-sim-4.1.0/"

## Go to location of the SDG script
SCRIPT_PATH="${PWD}/random_object.py"
OUTPUT="${PWD}/button_data"

## Go to Isaac Sim location for running with ./python.sh 
cd $ISAAC_SIM_PATH

echo "Starting Data Generation"  

./python.sh $SCRIPT_PATH --height 512 --width 512 --num_frames 20000 --data_dir $OUTPUT



