#!/bin/bash

SCRIPT="gp_symmetric.py"
PY_ALIAS="python3.10"

# Define what simulation you want to run {True, False}
GP=True
DP_COMPLEXITY=False

# Define the general result directory
GENERAL_RESULT_DIR="./results_real"

# Define the space of rounds of distillation to be simulated
MIN_DISTS=0
MAX_DISTS=5

# Define parameters as tuples of t_trunc, t_coh, p_gen, p_swap, w0, swaps
PARAMETER_SETS=(
    "-1 1400000 0.00092     0.85 0.952 2" # 50 km, 2 SWAP, rate is non-zero and higher when I distill
    # "-1 720000  0.0000150   0.85 0.867  1"  # 100 km, 1 SWAP, rate is 0 whatsoever
    # "-1 360000  0.000000096 0.85 0.36   0" # 200 km, 0 SWAP, hard to simulate
)

# -----------------------------------------
# GP: Parameter Set Testing with Various Optimizers and Spaces
# -----------------------------------------
# This section of the script is responsible for testing different parameter sets
# using various optimizers and search spaces. The goal is to evaluate the performance
# and effectiveness of each combination
#
# Follow these steps to run the script:
# 1. Define the parameter sets to be tested (above).
# 2. Specify the optimizers and search spaces to be used for testing (below).
#
# Note: Ensure that all necessary dependencies and environment variables are set
# before running this section of the script, and to choose the Python alias of your environment.
# -----------------------------------------

if [ "$GP" = "True" ]; then
    # Define the optimizers and spaces to test
    OPTIMIZER_SPACE_DP_COMBS=(
        "gp centerspace"
    )

    for PARAMETERS in "${PARAMETER_SETS[@]}"; do
        IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"
        
        T_COH="${PARAM_ARRAY[1]}"
        P_GEN="${PARAM_ARRAY[2]}"
        P_SWAP="${PARAM_ARRAY[3]}"
        W0="${PARAM_ARRAY[4]}"
        SWAPS="${PARAM_ARRAY[5]}"
        
        for TUPLE in "${OPTIMIZER_SPACE_DP_COMBS[@]}"; do
            OPTIMIZER=$(echo $TUPLE | awk '{print $1}')
            SPACE=$(echo $TUPLE | awk '{print $2}')
            DP_FLAG=$(echo $TUPLE | awk '{print $3}')
            
            FILENAME="output_${OPTIMIZER}_${SPACE}_${DP_FLAG}_tcoh${T_COH}_pgen${P_GEN}_pswap${P_SWAP}_w0${W0}.txt"
            TMPFILE=$(mktemp)
            
            echo "Running distillation with optimizer $OPTIMIZER and space $SPACE..."

            # Run the Python script with the specified parameters and append the output to TMPFILE
            { time $PY_ALIAS $SCRIPT \
                --min_swaps="$SWAPS" \
                --max_swaps="$SWAPS" \
                --min_dists="$MIN_DISTS" \
                --max_dists="$MAX_DISTS" \
                --optimizer="$OPTIMIZER" \
                --space="$SPACE" \
                --gp_shots=20 \
                --gp_initial_points=2 \
                --filename="$FILENAME" \
                --t_coh="$T_COH" \
                --p_gen="$P_GEN" \
                --p_swap="$P_SWAP" \
                --w0="$W0" \
                $DP_FLAG; } 2>&1 | tee -a "$TMPFILE"

            # Extract the time taken and append it to the output file
            echo "Time taken:" >> "$FILENAME"
            tail -n 3 "$TMPFILE" >> "$FILENAME"
            rm "$TMPFILE"

            # Create a folder for the results if it doesn't exist
            RESULT_DIR="$GENERAL_RESULT_DIR/results_${OPTIMIZER}_${SPACE}${DP_FLAG}"
            mkdir -p "$RESULT_DIR"

            # Move the output file to the results folder
            mv "$FILENAME" "$RESULT_DIR/"
            if ls *_${OPTIMIZER}.png 1> /dev/null 2>&1; then
                mv *_${OPTIMIZER}.png "$RESULT_DIR/"
            else
                echo "No plots yielded for optimizer $OPTIMIZER"
            fi
        done
    done

# -----------------------------------------
# DP: Parameter Set Testing with Various Optimizers and Spaces
# -----------------------------------------
# This section of the script is responsible for evaluating experimental complexity
# of simulations involving Memoization (dp).
#
# Again:
# 1. Define the parameter sets to be tested (above).
# 2. Specify the optimizers and search spaces to be used for testing (below).
# -----------------------------------------

elif [ "$DP_COMPLEXITY" = "True" ]; then
    START_DISTS=$MIN_DISTS
    LIMIT_DISTS=$MAX_DISTS
    GENERAL_RESULT_DIR="./results_dp_complexity"

    for MAX_DISTS in $(seq $START_DISTS $LIMIT_DISTS); do
        OPTIMIZER_SPACE_DP_COMBS=(
                "bf enumerate"
                "bf enumerate --dp"
            )
        for PARAMETERS in "${PARAMETER_SETS[@]}"; do
            IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"
            
            T_COH="${PARAM_ARRAY[1]}"
            P_GEN="${PARAM_ARRAY[2]}"
            P_SWAP="${PARAM_ARRAY[3]}"
            W0="${PARAM_ARRAY[4]}"
            SWAPS="${PARAM_ARRAY[5]}"

            for TUPLE in "${OPTIMIZER_SPACE_DP_COMBS[@]}"; do
                OPTIMIZER=$(echo $TUPLE | awk '{print $1}')
                SPACE=$(echo $TUPLE | awk '{print $2}')
                DP_FLAG=$(echo $TUPLE | awk '{print $3}')
                
                FILENAME="output_${OPTIMIZER}_${SPACE}${DP_FLAG}.txt"
                TMPFILE=$(mktemp)
                
                echo "Running distillation with optimizer $OPTIMIZER, space $SPACE and max_dists $MAX_DISTS..."

                # Run the Python script with the specified parameters and append the output to TMPFILE
                { time $PY_ALIAS $SCRIPT \
                    --min_swaps="$SWAPS" \
                    --max_swaps="$SWAPS" \
                    --min_dists="$MIN_DISTS" \
                    --max_dists="$MAX_DISTS" \
                    --optimizer="$OPTIMIZER" \
                    --space="$SPACE" \
                    --t_coh="$T_COH" \
                    --p_gen="$P_GEN" \
                    --p_swap="$P_SWAP" \
                    --w0="$W0" \
                    $DP_FLAG; } 2>&1 | tee -a "$TMPFILE"

                # Create a folder for the results if it doesn't exist
                RESULT_DIR="$GENERAL_RESULT_DIR/results_${OPTIMIZER}_${SPACE}${DP_FLAG}"
                mkdir -p "$RESULT_DIR"

                # Extract the time taken and append it to the output file
                real_time=$(tail -n 3 "$TMPFILE" | grep '^real' | awk '{print $2}')
                echo "swaps=$MAX_SWAPS, max_dists=$MAX_DISTS, time=$real_time" >> "$RESULT_DIR/$FILENAME"
                rm "$TMPFILE"
                for file in *_${OPTIMIZER}.png; 
                do 
                    mv "$file" "$RESULT_DIR/${MAX_DISTS}.png";
                done                
                mv output.txt "$RESULT_DIR/${MAX_DISTS}.txt"
            done
        done
    done
fi