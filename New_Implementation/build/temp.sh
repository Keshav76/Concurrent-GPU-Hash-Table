#!/bin/bash

# make
EXECUTABLE=./dynamic_test
INSERT_TRACE=/data/vipinpat/trace-files/insert_trace-400e7-100-add-no-dup.bin
SEARCH_TRACE=/data/vipinpat/trace-files/search_trace-400e7-30-find-no-dup-no-absent.bin
DELETE_TRACE=/data/vipinpat/trace-files/delete_trace-400e7-10-rem-no-dup-no-absent.bin

# Arguments
ARG1=30000000
ARG2=2
ARG3=100000
ARG4=0.5
ARG5=0.85
ARG6=0.85

for i in {1..10}
do
    echo "Starting execution $i"
    echo $EXECUTABLE $INSERT_TRACE $SEARCH_TRACE $DELETE_TRACE $ARG1 $ARG2 $ARG3 $ARG4 $ARG5 $ARG6
done