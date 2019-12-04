# CUDA-DLA-fractal-gpu
Implementation of the diffusion limited aggregation fractal with CUDA C++

# Building and Running
## Prerequisits
This project depends on:
* Rust
  * Install Rustup/Cargo here: https://www.rust-lang.org/
* CUDA
  * This requires an NVIDIA gpu with support for CUDA. See https://developer.nvidia.com/cuda-gpus to see if your GPU is supported. I tested this on an RTX 2070.
  * It also requires the CUDA runtime/drivers to be installed. See https://developer.nvidia.com/cuda-downloads. This was tested in V10.0.130.

There are also prebuild binaries for Linux in the bin folder. These might not be up-to-date with the latest version. These still require CUDA runtime/drivers to run, but no Rust.

These instructions assume you are running Linux.

## Get this repository
Move to the folder where you want this project do be located and run in the terminal:
```git clone https://github.com/aeduin/CUDA-DLA-fractal-gpu.git```

## Running prebuild binaries
`cd` into the bin folder. Mark the two files as executable (`chmod +x <path_to_file_goes_here>`, for each file). After that, first run `./diffusion-limited-aggregation`, which produces a grid_output.bin. Then `cd` into the rust-visualiser folder and run `./rust-visualiser` to turn the grid_output.bin into an image named image.png.

## Compiling yourself
Use the script `./compile-run` to run the fractal generator that outputs the grid_output.bin file. `cd` into the rust-visualiser folder and run `cargo run --release` to turn the grid_output.bin into an image named image.png.