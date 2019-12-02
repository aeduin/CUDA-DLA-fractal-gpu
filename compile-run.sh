#!/bin/bash

# output="$(nvcc main.cu)"

nvcc main.cu -o diffusion-limited-aggregation && ./diffusion-limited-aggregation

# echo $output
