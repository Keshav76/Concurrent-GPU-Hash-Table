#!/bin/bash

# Configurations (mapped from label to numeric value)
declare -A config_map=(
    ["1e7"]="10000000"
    ["5e7"]="50000000"
    ["8e7"]="80000000"
    ["1e8"]="100000000"
    ["5e8"]="500000000"
    ["8e8"]="800000000"
)

# Trace groups
trace_groups=(
"insert_trace-400e7-100-add-no-dup.bin search_trace-400e7-30-find-no-dup-no-absent.bin delete_trace-400e7-10-rem-no-dup-no-absent.bin"
"insert_trace-500e7-50-add-10-dup.bin search_trace-500e7-40-find-20-dup-no-absent.bin delete_trace-500e7-10-rem-40-dup-no-absent.bin"
"insert_trace-400e7-50-add-20-dup.bin search_trace-400e7-40-find-40-dup-10-absent.bin delete_trace-400e7-10-rem-50-dup-20-absent.bin"
)

# Output root folders
roots=("first" "second" "third")

# Loop through trace groups
for i in "${!trace_groups[@]}"; do
    group="${trace_groups[$i]}"
    root_dir="./Output/${roots[$i]}"
    mkdir -p "$root_dir"

    for label in "${!config_map[@]}"; do
        size="${config_map[$label]}"
        echo "=== Running for $root_dir with size: $size ==="

        output_dir="$root_dir/$size"
        mkdir -p "$output_dir"

        read -r insert_file search_file delete_file <<< "$group"

        # Extract raw numbers from filenames
        ins_raw=$(echo "$insert_file" | cut -d- -f3)
        lkp_raw=$(echo "$search_file" | cut -d- -f3)
        rem_raw=$(echo "$delete_file" | cut -d- -f3)

        # Convert to flo  at (percent → fraction)
        ins_val=$(echo "scale=2; $ins_raw / 100" | bc)

        ./script.sh /data/vipinpat/trace-files/$insert_file /data/vipinpat/trace-files/$search_file /data/vipinpat/trace-files/$delete_file $size > "output_New_Implementation.txt"

        # Move output files
        for f in "output_New_Implementation.txt"; do
            if [ -f "$f" ]; then
                mv "$f" "$output_dir/"
            else
                echo "Warning: $f not found after abc.sh run."
            fi
        done
    done
done
