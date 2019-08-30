#!/bin/bash

# output="$(nvcc main.cu)"

if !(nvcc main.cu -o diffusion-limited-aggregation 2>&1 | grep "error") ; then
./diffusion-limited-aggregation 
fi

# echo $output
