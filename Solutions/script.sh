#!/bin/bash

# Check if parameters are passed
if [ $# -eq 0 ]; then
    echo "Usage: $0 --size=VALUE --ins=VALUE --lkp=VALUE --rem=VALUE --insert_path=VALUE --delete_path=VALUE --search_path=VALUE"
    exit 1
fi

# Store the parameters
params="$@"

# Loop through all directories in the current directory
for dir in "Better GPU" "SlabHash" ; do
    dir=${dir%/}  # Remove trailing slash

    # echo "Processing directory: $dir"

    # Move into the directory
    cd "$dir" || { echo "Failed to enter directory $dir"; continue; }

    # Compile exp.cu
    echo "Skip Compiling $dir..."
    # nvcc -ccbin /usr/bin/gcc-10 -std=c++17 exp.cu -o output -lcudart -lstdc++ -arch=sm_60 --expt-relaxed-constexpr -lm -w

    # Check if compilation was successful
    if [ ! -f "./output" ]; then
        echo "Compilation failed in $dir, skipping..."
        cd ..
        continue
    fi

    # Run ./output 5 times and save output
    echo "Running $dir..."
    output_file="../output_${dir}.txt"  # Store output in the parent directory
    > "$output_file"  # Clear previous content

    for i in {1..5}; do
        echo "Run $i:" >> "$output_file"
        ./output $params --verbose=0 >> "$output_file"  # Pass the user-specified parameters
        echo "" >> "$output_file"
    done

    # Move back to the parent directory
    cd ..
done

echo "Processing completed!"
