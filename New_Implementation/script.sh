#!/bin/bash
params="$@"


ARG2=2
ARG3=100000
ARG4=0.5
ARG5=0.85
ARG6=0.85

cd build

for i in {1..5}
do
    echo "Starting execution $i"
    ./dynamic_test $params $ARG2 $ARG3 $ARG4 $ARG5 $ARG6
done

cd ..