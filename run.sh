#!/bin/bash

# Run with different hyperparams
pip install .
OUT_DIR=$(ls -d output/ray_results/* | tail -n 1)
python src/atari_cr/hyperparams.py & sleep 5 && tensorboard --logdir "$OUT_DIR"

# Copy assets from /tmp/ray
# Function to find the only subdirectory in the current directory
find_only_subdir() {
    local dir="$1"
    local subdir=$(find "$dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -z "$subdir" ]; then
        echo "No subdirectory found in $dir"
        exit 1
    fi
    echo "$subdir"
}
copy_assets() {
    local input_path="$1"
    
    # Navigate to the artifacts directory
    cd /tmp/ray/session_latest/artifacts || { echo "Artifacts directory not found"; exit 1; }

    # Find the first subdirectory
    SUBDIR=$(find_only_subdir ".")
    cd "$SUBDIR" || exit 1

    # Find the second subdirectory
    INNER_SUBDIR=$(find_only_subdir ".")

    FULL_PATH="$(realpath "$INNER_SUBDIR")/working_dirs"

    # Iterate over elements in working_dirs
    for ELEMENT in "$FULL_PATH"/*; do
        if [ -d "$ELEMENT" ]; then
            NAME=$(basename "$ELEMENT" | cut -c1-11)
            TARGET_DIR=$(find "$input_path" -mindepth 1 -maxdepth 1 -type d -name "$NAME*")
            
            if [ -d "$TARGET_DIR" ]; then
                # Locate the tuning directory
                TUNING_SUBDIR=$(find "$ELEMENT/output/runs/tuning" -mindepth 1 -maxdepth 1 -type d | head -n 1)
                
                if [ -d "$TUNING_SUBDIR" ]; then
                    cp -r "$TUNING_SUBDIR"/* "$TARGET_DIR"/
                    echo "Copied contents of $TUNING_SUBDIR to $TARGET_DIR"
                else
                    echo "No tuning subdirectory found in $ELEMENT"
                fi
            else
                echo "No target directory found for $NAME in $input_path"
            fi
        fi
    done
}
copy_assets "$OUT_DIR"