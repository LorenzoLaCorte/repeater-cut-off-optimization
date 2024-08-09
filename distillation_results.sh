#!/bin/bash

# Define what simulation you want to run
ALTERNATE=False
ONE_LEVEL=False
GP=True

# Define the general result directory
GENERAL_RESULT_DIR="./results"

MAX_SWAPS=4
MAX_DISTS=10

# Tuples of t_trunc, t_coh, p_gen, p_swap, w0
PARAMETER_SETS=(
    "1000 400 0.9 0.9 0.933"
)

if [ "$ALTERNATE" = "True" ]; then
    for PARAMETERS in "${PARAMETER_SETS[@]}"; do
        IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"
        
        python distillation_alternate.py \
            --t_trunc "${PARAM_ARRAY[0]}" \
            --t_coh "${PARAM_ARRAY[1]}" \
            --p_gen "${PARAM_ARRAY[2]}" \
            --p_swap "${PARAM_ARRAY[3]}" \
            --w0 "${PARAM_ARRAY[4]}"
    done

elif [ "$ONE_LEVEL" = "True" ]; then
    exit # TODO: implement

elif [ "$GP" = "True" ]; then
    OPTIMIZER_SPACE_DP_COMBS=(
        "bf one_level"
        "gp one_level"
    )
    for PARAMETERS in "${PARAMETER_SETS[@]}"; do
        IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAMETERS"
        
        T_COH="${PARAM_ARRAY[1]}"
        P_GEN="${PARAM_ARRAY[2]}"
        P_SWAP="${PARAM_ARRAY[3]}"
        W0="${PARAM_ARRAY[4]}"
        
        for TUPLE in "${OPTIMIZER_SPACE_DP_COMBS[@]}"; do
            OPTIMIZER=$(echo $TUPLE | awk '{print $1}')
            SPACE=$(echo $TUPLE | awk '{print $2}')
            DP_FLAG=$(echo $TUPLE | awk '{print $3}')
            
            FILENAME="output_${OPTIMIZER}_${SPACE}_${DP_FLAG}_tcoh${T_COH}_pgen${P_GEN}_pswap${P_SWAP}_w0${W0}.txt"
            TMPFILE=$(mktemp)
            
            echo "Running distillation with optimizer $OPTIMIZER and space $SPACE..."

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
    done
fi