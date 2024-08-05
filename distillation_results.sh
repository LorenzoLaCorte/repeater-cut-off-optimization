#!/bin/bash

MAX_SWAPS=3
MAX_DISTS=7
OPTIMIZER_SPACE_COMBS=(
    "gp strategy 24 0.002 0.5 1"
    "bf enumerate 24 0.002 0.5 1"

    "gp strategy 24 0.002 0.5 0.999"
    "bf enumerate 24 0.002 0.5 0.999"
)

for TUPLE in "${OPTIMIZER_SPACE_COMBS[@]}"; do
    OPTIMIZER=$(echo $TUPLE | awk '{print $1}')
    SPACE=$(echo $TUPLE | awk '{print $2}')
    T_COH=$(echo $TUPLE | awk '{print $3}')
    P_GEN=$(echo $TUPLE | awk '{print $4}')
    P_SWAP=$(echo $TUPLE | awk '{print $5}')
    W0=$(echo $TUPLE | awk '{print $6}')
    
    FILENAME="output_${OPTIMIZER}_${SPACE}_tcoh${T_COH}_pgen${P_GEN}_pswap${P_SWAP}_w0${W0}.txt"
    TMPFILE=$(mktemp)
    
    echo "Running distillation with optimizer $OPTIMIZER and space $SPACE..."

    { time python distillation_gp.py \
        --max_swaps="$MAX_SWAPS" \
        --max_dists="$MAX_DISTS" \
        --optimizer="$OPTIMIZER" \
        --space="$SPACE" \
        --filename="$FILENAME" \
        --t_coh "${T_COH[@]}" \
        --p_gen "${P_GEN[@]}" \
        --p_swap "${P_SWAP[@]}" \
        --w0 "${W0[@]}"; } 2>&1 | tee -a "$TMPFILE"

    # Extract the time taken and append it to the output file
    echo "Time taken:" >> "$FILENAME"
    tail -n 3 "$TMPFILE" >> "$FILENAME"

    rm "$TMPFILE"
done