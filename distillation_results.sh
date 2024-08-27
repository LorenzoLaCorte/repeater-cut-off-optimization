#!/bin/bash

# Define what simulation you want to run
ALTERNATE=False
ONE_LEVEL=False
GP=True
DP_COMPLEXITY=False

# Define the general result directory
GENERAL_RESULT_DIR="./results_cmp"

# MIN_SWAPS=1
# MAX_SWAPS=3
MIN_DISTS=0
MAX_DISTS=5

# Tuples of t_trunc, t_coh, p_gen, p_swap, w0, swaps
PARAMETER_SETS=(
    "-1 1440000 0.0026 0.85 0.9577 3" # 20 km, 3 SWAP, rate is non-zero only with dist. and optimal for k=4
    "-1 720000 0.0009187 0.85 0.9524 2" # 50 km, 2 SWAP, rate is non-zero and higher when I distill
    "-1 360000 0.0000150 0.85 0.867 1"  # 100 km, 1 SWAP, rate is 0 whatsoever
    # "-1 180000 0.000000096 0.85 0.36" # 200 km, 0 SWAP, hard to simulate
    # -----------------------------------------
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
        "bf enumerate"
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
            { time python3.10 distillation_gp.py \
                --min_swaps="$SWAPS" \
                --max_swaps="$SWAPS" \
                --min_dists="$MIN_DISTS" \
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

elif [ "$DP_COMPLEXITY" = "True" ]; then
    MIN_SWAPS=$MAX_SWAPS # Only one value for swaps
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
            
            for TUPLE in "${OPTIMIZER_SPACE_DP_COMBS[@]}"; do
                OPTIMIZER=$(echo $TUPLE | awk '{print $1}')
                SPACE=$(echo $TUPLE | awk '{print $2}')
                DP_FLAG=$(echo $TUPLE | awk '{print $3}')
                
                FILENAME="output_${OPTIMIZER}_${SPACE}${DP_FLAG}.txt"
                TMPFILE=$(mktemp)
                
                echo "Running distillation with optimizer $OPTIMIZER, space $SPACE and max_dists $MAX_DISTS..."

                # Run the Python script with the specified parameters and append the output to TMPFILE
                { time python distillation_gp.py \
                    --min_swaps="$MIN_SWAPS" \
                    --max_swaps="$MAX_SWAPS" \
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