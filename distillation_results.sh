#!/bin/bash

MAX_SWAPS=3
MAX_DISTS=7
OPTIMIZER_SPACE_COMBS=(
    "bf enumerate 10000 0.01 0.5 0.98 False"
    "gp strategy 10000 0.01 0.5 0.98 False"
    "bf enumerate 10000 0.01 0.5 0.98 True"
    "gp strategy 10000 0.01 0.5 0.98 True"
)

# Create a general results folder
TODAY=$(date +%F)
GENERAL_RESULT_DIR="results_$TODAY"
mkdir -p "$GENERAL_RESULT_DIR"

for TUPLE in "${OPTIMIZER_SPACE_COMBS[@]}"; do
    OPTIMIZER=$(echo $TUPLE | awk '{print $1}')
    SPACE=$(echo $TUPLE | awk '{print $2}')
    T_COH=$(echo $TUPLE | awk '{print $3}')
    P_GEN=$(echo $TUPLE | awk '{print $4}')
    P_SWAP=$(echo $TUPLE | awk '{print $5}')
    W0=$(echo $TUPLE | awk '{print $6}')
    DP=$(echo $TUPLE | awk '{print $7}')
    
    FILENAME="output_${OPTIMIZER}_${SPACE}_dp=${DP}_tcoh${T_COH}_pgen${P_GEN}_pswap${P_SWAP}_w0${W0}.txt"
    TMPFILE=$(mktemp)
    
    # Check if DP is set to "True" and set the DP_FLAG accordingly
    if [ "$DP" = "True" ]; then
        DP_FLAG="--dp"
    else
        DP_FLAG=""
    fi

    echo "Running distillation with optimizer $OPTIMIZER, space $SPACE, and DP=$DP..."

    # Run the Python script with the specified parameters and append the output to TMPFILE
    { time python distillation_gp.py \
        --max_swaps="$MAX_SWAPS" \
        --max_dists="$MAX_DISTS" \
        --optimizer="$OPTIMIZER" \
        --space="$SPACE" \
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
    mv *_${OPTIMIZER}.png "$RESULT_DIR/"
done