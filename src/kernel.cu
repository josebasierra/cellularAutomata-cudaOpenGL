#include "kernel.cuh"
#include <math.h>

__global__ void kernel(Particle *particles, int N, int deltaTime)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        float speed = 20.0f;
        particles[i].x += speed*deltaTime/1000.f;
    }

}


void run_kernel(Particle *particles, int N, int deltaTime)
{
    dim3 block(32, 1, 1);
    dim3 grid(N/block.x, 1, 1);
    kernel<<< grid, block>>>(particles, N, deltaTime);
}
