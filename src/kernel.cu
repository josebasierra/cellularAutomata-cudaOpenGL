#include "kernel.cuh"
#include <math.h>


__global__ void updateCellularState(bool* input, bool* output, int SIZE)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (x < SIZE && y < SIZE) {
        output[x*SIZE + y] = !input[x*SIZE + y];
    }

}


void cuda_updateCellularState(bool* input, bool* output, int SIZE) {
    dim3 block(32, 32, 1);
    dim3 grid(SIZE/block.x, SIZE/block.y, 1);
    updateCellularState<<<grid, block>>>(input, output, SIZE);
}


