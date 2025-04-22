


#!/bin/bash

# Path to your binary and trace files
EXECUTABLE=./dynamic_test
INSERT_TRACE=insert_trace-400e7-50-add-10-dup.bin
SEARCH_TRACE=search_trace-400e7-40-find-40-dup-10-absent.bin
DELETE_TRACE=delete_trace-400e7-10-rem-50-dup-20-absent.bin

# Arguments
ARG1=300000000
ARG2=2
ARG3=300000000
ARG4=0.5
ARG5=0.85
ARG6=0.85

# Loop for 10 iterations
for i in {1..5}
do
    echo "Running iteration $i..."
    $EXECUTABLE $INSERT_TRACE $SEARCH_TRACE $DELETE_TRACE $ARG1 $ARG2 $ARG3 $ARG4 $ARG5 $ARG6
    echo "Iteration $i completed."
    echo "-----------------------------"
done
