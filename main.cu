#include <iostream>
#include <math.h>
#include <fstream>

// Set CIRCLE_BORDER to a negative value to activate normal mode instead
#define CIRCLE_BORDER -1

// If RANDOM_WALK is defined, the direction of a particle changes every tick
#define RANDOM_WALK

typedef struct {
    float x;
    float y;
    float horizontal_speed;
    float vertical_speed;
    uint seed;
} Particle;

using ullong = unsigned long long;

const float radius = 2.0f;                  // Size of a single particle
const int ceil_radius = (int)radius + ((((float)(int)radius) < radius) ? 1 : 0);    // Radius rounded up
const float max_speed = 3.0f;               // Maximum speed a particle can have
const int particle_count = 4096 * 64;       // The amount of particles simulated at once, should be a multiple of particle_threads_per_block
const int particle_threads_per_block = 16;  // Amount of threads in one thread block when calculating a tick, and when initializing partickles

const int grid_size = 1024 * 4;             // Size of the grid on which the fractal is generated
const int grid_width = grid_size;
const int grid_height = grid_size;

__device__ int grid[grid_height][grid_width];   // The grid on which the fractal will be generated

// Information about the boundaries in which all static particles are
__device__ int border_left;
__device__ int border_right;
__device__ int border_top;
__device__ int border_bottom;
__device__ int smallest_distance_to_center;

// Work in progress
__device__ ullong total_static_particles;
__device__ ullong weight_center_x;
__device__ ullong weight_center_y;

// Used in debugging
const int debug_array_size = 1024;
__device__ int debug = 0;
__device__ int debug_array[debug_array_size];

void VecAdd();
void simulate();
void tick(Particle* particles, int tick_count);
__host__ __device__ int random_int(int min, int max, uint seed);
__device__ Particle make_static(Particle particle, int tick_count, float modulo_x, float modulo_y);
__host__ __device__ float random_float(uint seed);

#define print(message) std::cout << message << std::endl

/*
    Entry point: main
*/
int main() {
    print("starting");
    simulate();
    print("done");
}

// checks for cuda errors
// could be improved
void cuda_error() {
    auto result = cudaGetLastError();
    if (result != cudaSuccess) {
        do {
            std::cout << "error: " << result << std::endl;
            std::cout << "error message: " << cudaGetErrorString(result) << std::endl;

            result = cudaGetLastError();
            break;
        }
        while(result != cudaSuccess);
    }
    else {
        std::cout << "success" << std::endl;
    }
}

// Sets all the grid values to -1 (meaning 'empty' or 'no static particle here')
__global__ void init_grid_negative() {
    grid[blockIdx.y * blockDim.y + threadIdx.y][blockIdx.x * blockDim.x + threadIdx.x] = -1;
}

// Performs all the initialization that happens only once on the gpu
__global__ void init_gpu_single() {
    border_top = grid_height / 2;
    border_bottom = grid_height / 2;
    border_left = grid_width / 2;
    border_right = grid_width / 2;
    smallest_distance_to_center = CIRCLE_BORDER * CIRCLE_BORDER;

    if(CIRCLE_BORDER < 0) {
        // Set the center of the grid to 0 (meaning there is a static particle in the center)
        grid[grid_height / 2][grid_width / 2] = 0;
    }
    else {
        // Init weight center
        int center_bias = 10;
        total_static_particles = center_bias;
        weight_center_x = (grid_width / 2) * center_bias;
        weight_center_y = (grid_height / 2) * center_bias;
    }
}

// Outputs the grid (preceded by its width/height) to a file
void output_grid() {
    // Get grid from GPU memory
    size_t mem_size = sizeof(int) * grid_height * grid_width;
    int* host_grid = (int*)malloc(mem_size);
    cudaMemcpyFromSymbol(host_grid, grid, mem_size, 0, cudaMemcpyDeviceToHost);

    // Create file
    std::ofstream output_file;
    output_file.open("grid_output.bin", std::ios::binary);
    if(output_file.is_open()) {
        print("output_file is open");
    }

    // Output to file
    const int grid_size[2] = {grid_width, grid_height};
    output_file.write((const char*) &grid_size, sizeof(int) * 2);
    output_file.write((const char*) host_grid, mem_size);
    
    // clean up
    output_file.close();
    delete host_grid;
}

// Returns a pseudorandom number based on the input number x
__host__ __device__ uint hash(uint x) {
    const uint seed = 1324567967;
    x += seed;
    x = ((x >> 16) ^ x) * seed;
    x = ((x >> 16) ^ x) * seed;
    x = (x >> 16) ^ x;
    return x;
}

// Returns an int in the range [min, max) based on seed
__host__ __device__ int random_int(int min, int max, uint seed) {
    uint random = hash(seed);
    random %= (uint)(max - min);
    
    return (int)random + min;
}

// Returns a float in the range [0, 1) based on seed;
__host__ __device__ float random_float(uint seed) {
    const int max = 10000000;
    int base = random_int(0, max, seed);

    return (float)base / (float)max;
}

// Randomizes the speed and direction of a particle
__device__ Particle randomize_speed(Particle particle, int direction_seed, int speed_seed) {
    float direction = M_PI * 2.0f * random_float(direction_seed);
    float speed = random_float(speed_seed) * max_speed;

    particle.vertical_speed = cosf(direction) * speed;
    particle.horizontal_speed = sinf(direction) * speed;

    return particle;
}

// Randomizes all fields of the particle
__device__ Particle randomize_particle(Particle particle) {
    uint seed = particle.seed;
    int center_height = border_bottom - border_top;

    if(CIRCLE_BORDER < 0) {
        // Place the particle outside the borders
        particle.x = random_int(0, grid_width, seed + 0);

        if(particle.x > border_left && particle.x < border_right) {
            // Generate a y-coordiante that does not overlap with the borders
            particle.y = random_int(0, grid_height - center_height, seed + 1);

            if(particle.y > border_top) {
                particle.y += center_height;
            }
        }
        else {
            particle.y = random_int(0, grid_height, seed + 1);
        }
    }
    else {
        particle.x = (float) (grid_width - (weight_center_x / total_static_particles));
        particle.y = (float) (grid_height - (weight_center_y / total_static_particles));
    }

    #ifndef RANDOM_WALK
    // When doing a random walk, the speed is randomized at the beginning of each tick anyways
    particle = randomize_speed(particle, seed + 2, seed + 3);
    #endif

    particle.seed = hash(seed);

    return particle;
}

// Initializes the particle
__global__ void init_particles(Particle* particles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // particle index in the particles array
    Particle* particle = particles + i;
    particle->seed = (uint)i * 4;
    *particle = randomize_particle(*particle);
}

// Prints border_left, border_right, border_top and border_bottom to stdio
void print_boundaries() {
    int left, right, top, bottom;

    cudaMemcpyFromSymbol(&left, border_left, sizeof(int));
    cudaMemcpyFromSymbol(&right, border_right, sizeof(int));
    cudaMemcpyFromSymbol(&top, border_top, sizeof(int));
    cudaMemcpyFromSymbol(&bottom, border_bottom, sizeof(int));
    print(left << ", " << right << ", " << top << ", " << bottom);
}

// Creates the fractal
void simulate() {
    // Make sure there are no errors at the start of the simulation
    cuda_error();

    // Initialize grid
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(grid_width / threadsPerBlock.x, grid_height / threadsPerBlock.y);
    init_grid_negative<<<blocks, threadsPerBlock>>>();
    init_gpu_single<<<1, 1>>>();

    // Initialize particles
    size_t mem_size = particle_count * sizeof(Particle);
    Particle* particles;
    cudaMalloc(&particles, mem_size);
    const int particle_blocks = particle_count / particle_threads_per_block;

    cuda_error();
    init_particles<<<particle_blocks, particle_threads_per_block>>>(particles);
    // Done intializing particles

    // Print some debug information, I left this in since it safed me some confusion a few times
    print_boundaries();
    cuda_error();

    // Perform simulation ticks, until a particle hits the margins
    int tick_count = 0;
    for(int i = 0; true; i++) {
        // Perform one tick
        tick(particles, ++tick_count);

        // Debug information
        int left, right, top, bottom, center_distance;
        int debug_copy;
        int debug_array_copy[debug_array_size];
        ullong total_static_particles_copy;

        cudaMemcpyFromSymbol(&left, border_left, sizeof(int));
        cudaMemcpyFromSymbol(&right, border_right, sizeof(int));
        cudaMemcpyFromSymbol(&top, border_top, sizeof(int));
        cudaMemcpyFromSymbol(&bottom, border_bottom, sizeof(int));
        cudaMemcpyFromSymbol(&bottom, border_bottom, sizeof(int));
        cudaMemcpyFromSymbol(&center_distance, smallest_distance_to_center, sizeof(int));
        cudaMemcpyFromSymbol(&debug_copy, debug, sizeof(int));
        cudaMemcpyFromSymbol(&total_static_particles_copy, total_static_particles, sizeof(ullong));

        cudaMemcpyFromSymbol(&debug_array_copy, debug_array, sizeof(int) * debug_copy);

        if(i % 10000 == 0) {
            print(left << ", " << right << ", " << top << ", " << bottom << ", " << center_distance);
            print(debug_copy);
            // print(total_static_particles_copy);

            for(int i = 0; i < debug_copy && i < 1024; i++) {
                if(i % 2 == 0) {
                    print("");
                }
                print(debug_array_copy[i]);
            }
        }

        // Check if a particle has come to close to the margins, and if it did, finish the simulation
        const int margin = 150;
        if(CIRCLE_BORDER > -1 && center_distance < margin * margin) {
            break;
        }
        if(left < margin || right > grid_width - margin || top < margin || bottom > grid_height - margin) {
            break;
        }
    }

    // Finish up
    cuda_error();
    output_grid();

    cudaFree(particles);
}

__device__ float pythagoras(float a, float b) {
    return a * a + b * b;
}

__device__ float pythagoras(Particle particle) {
    return pythagoras(particle.x - (float)(grid_width / 2), particle.y - (float)(grid_height / 2));
}

__device__ Particle move_particle(Particle particle) {
    
    #ifdef RANDOM_WALK
    // randomize direction and speed
    particle = randomize_speed(particle, particle.seed, particle.seed + 1);
    particle.seed = hash(particle.seed);
    #endif

    // move particle
    particle.x += particle.horizontal_speed;
    particle.y += particle.vertical_speed;

    // check bounds
    if(particle.x - radius <= 0.0f) {
        particle.x = 0.01f + radius;
        particle.horizontal_speed *= -1.0f;
    }
    else if(particle.x + radius >= grid_width) {
        particle.x = grid_width - 0.01f - radius;
        particle.horizontal_speed *= -1.0f;
    }
    if(particle.y - radius <= 0.0f) {
        particle.y = 0.01f + radius ;
        particle.vertical_speed *= -1.0f;
    }
    else if(particle.y + radius >= grid_height) {
        particle.y = grid_height - 0.01f - radius;
        particle.vertical_speed *= -1.0f;
    }

    return particle;
}

// Performs one step of one particle
__global__ void particle_step(Particle* particles, int tick_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index in the particles array
    Particle particle = particles[i];

    const int max_steps = 5;
    bool outside_border_margins = true; // Move at least once
    const int border_margins = 150;

    if(CIRCLE_BORDER < 0) {
        // Perform at most max_steps when outside the border margins
        for(int i = 0; i < max_steps && outside_border_margins; i++) {
            particle = move_particle(particle);
            outside_border_margins = particle.x < (border_left - border_margins) || particle.x > (border_right + border_margins) || particle.y > (border_bottom + border_margins) || particle.y < (border_top - border_margins);
        }
    }
    else {
        // set to false to avoid confusion
        outside_border_margins = false;
        particle = move_particle(particle);
    }

    float modulo_x = fmod(particle.x, 1.0f);
    float modulo_y = fmod(particle.y, 1.0f);

    // If the particle is outside the circle border, turn it static
    if(CIRCLE_BORDER > -1 && (int)(pythagoras(particle) + radius) >= CIRCLE_BORDER * CIRCLE_BORDER) {
        particles[i] = make_static(particle, tick_count, modulo_x, modulo_y);
        return;
    }

    if(!outside_border_margins) {
        // Check for collision with static particles
        bool looping = true;
        
        for(int dx = -ceil_radius; dx <= ceil_radius && looping; dx++) {
            for(int dy = -ceil_radius; dy <= ceil_radius && looping; dy++) {
                // Calculate distance from center of the particle
                float distance_x = -dx + modulo_x;
                float distance_y = -dy + modulo_y;

                if(pythagoras(distance_x, distance_y) < radius * radius) {
                    // Position is within distance of the center
                    
                    if(grid[(int)(particle.y - distance_y)][(int)(particle.x - distance_x)] >= 0) {
                        // It hit another particle, so turn this one static too
                        particle = make_static(particle, tick_count, modulo_x, modulo_y);
    
                        looping = false;
                        break;
                    }
                }
            }
        }
    }

    // Update value in particles array
    particles[i] = particle;
}

// Creates a static particle in the grid on this location, and replaces the live particle by a new one
__device__ Particle make_static(Particle particle, int tick_count, float modulo_x, float modulo_y) {
    // Create new static particle
    for(int dx = -ceil_radius; dx <= ceil_radius; dx++) {
        for(int dy = -ceil_radius; dy <= ceil_radius; dy++) {
            // Calculate distance from center of the particle
            float distance_x = -dx + modulo_x;
            float distance_y = -dy + modulo_y;

            if(distance_x * distance_x + distance_y * distance_y < radius * radius) {
                // Calculate position in grid
                int absolute_x = (int)(particle.x - distance_x);
                int absolute_y = (int)(particle.y - distance_y);
                
                // If the absolute_x/y are within the grid
                if(absolute_x >= 0 && absolute_x < grid_width && absolute_y >= 0 && absolute_y < grid_height) {
                    // Set the grid to being hit
                    grid[absolute_y][absolute_x] = tick_count;

                    /*
                        Because the program writes and reads from the same grid in a single tick,
                        the algorithm isn't completely deterministic. I could use two different 
                        grids and then copy values, but it doesn't feel necessary.
                    */
                }
            }
        }
    }

    // Update information about the boundaries that contain all particles
    if(CIRCLE_BORDER < 0) {
        atomicMin(&border_left, (int)(particle.x - radius));
        atomicMax(&border_right, (int)(particle.x + radius));
        atomicMin(&border_top, (int)(particle.y - radius));
        atomicMax(&border_bottom, (int)(particle.y + radius));
    }
    else {
        atomicMin(&smallest_distance_to_center, (int)(pythagoras(particle) - radius));
        atomicAdd(&total_static_particles, 1l);
        atomicAdd(&weight_center_x, (ullong)particle.x);
        atomicAdd(&weight_center_y, (ullong)particle.y);
    }

    // Give the particle a random new position and speed
    return randomize_particle(particle);
}

// Performs one tick
void tick(Particle* particles, int tick_count) {
    const int blocks = particle_count / particle_threads_per_block;

    particle_step<<<blocks, particle_threads_per_block>>>(particles, tick_count);
}
