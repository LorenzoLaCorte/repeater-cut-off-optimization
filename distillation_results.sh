#!/bin/bash

MAX_SWAPS=3
MAX_DISTS=7
OPTIMIZER_SPACE_COMBS=(
    "bf enumerate 120 0.9 0.9 0.933 False"
    "gp strategy 120 0.9 0.9 0.933 False"
    "bf enumerate 120 0.9 0.9 0.933 True"
    "gp strategy 120 0.9 0.9 0.933 True"
)

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
done