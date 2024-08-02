#!/bin/bash

MAX_SWAPS=3
MAX_DISTS=7
OPTIMIZER="gp"
SPACE="enumerate"
FILENAME="output_${OPTIMIZER}_${SPACE}.txt"
TMPFILE=$(mktemp)

{ time py distillation_gp.py \
    --max_swaps="$MAX_SWAPS" \
    --max_dists="$MAX_DISTS" \
    --optimizer="$OPTIMIZER" \
    --space="$SPACE" \
    --filename="$FILENAME"; } 2>> "$TMPFILE"

# Extract the time taken and append it to the output file
echo "Time taken:" >> "$FILENAME"
tail -n 3 "$TMPFILE" >> "$FILENAME"

rm "$TMPFILE"