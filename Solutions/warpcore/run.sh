#!/bin/bash

# Path to your binary and trace files
EXECUTABLE=./basic_usage_from_host
INSERT_TRACE=../../../../vipinpat/trace-files/insert_trace-400e7-50-add-10-dup.bin
SEARCH_TRACE=../../../../vipinpat/trace-files/search_trace-400e7-40-find-40-dup-10-absent.bin
DELETE_TRACE=../../../../vipinpat/trace-files/delete_trace-400e7-10-rem-50-dup-20-absent.bin

# Arguments
ARG1=50000000

# Loop for 10 iterations
for i in {1..5}
do
    echo "Running iteration $i..."
     $EXECUTABLE $INSERT_TRACE $SEARCH_TRACE $DELETE_TRACE $ARG1
    echo "Iteration $i completed."
    echo "-----------------------------"
done
