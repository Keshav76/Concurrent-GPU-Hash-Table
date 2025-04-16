


#!/bin/bash

# Path to your binary and trace files
EXECUTABLE=./dynamic_test
INSERT_TRACE=insert_trace-400e7-100-add-no-dup.bin
SEARCH_TRACE=search_trace-400e7-30-find-no-dup-no-absent.bin
DELETE_TRACE=delete_trace-400e7-10-rem-no-dup-no-absent.bin

# Arguments
ARG1=100000000
ARG2=2
ARG3=1000000
ARG4=0.5
ARG5=0.85
ARG6=0.85

# Loop for 10 iterations
for i in {1..5}
do
    echo "Running iteration $i..."
    ncu --target-processes all $EXECUTABLE $INSERT_TRACE $SEARCH_TRACE $DELETE_TRACE $ARG1 $ARG2 $ARG3 $ARG4 $ARG5 $ARG6
    echo "Iteration $i completed."
    echo "-----------------------------"
done
