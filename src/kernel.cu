#include "kernel.cuh"
#include <stdio.h>
#include <math.h>


__global__ void updateCellularState(bool* input, bool* output, bool* binary_rule, int grid_size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    int width = grid_size;
    int height = grid_size;

    if (x < width && y < height)
    {    
        int xRight = (x + 1 + width) % width;
        int xLeft = (x - 1 + width) % width ;
        int yTop = (y + 1 + height) % height; 
        int yBottom = (y - 1 + height) % height;

        bool neigh0 = input[xLeft*height + yTop]; 
        bool neigh1 = input[x*height + yTop]; 
        bool neigh2 = input[xRight*height + yTop]; 
        bool neigh3 = input[xLeft*height + y];
        bool neigh4 = input[xRight*height + y];
        bool neigh5 = input[xLeft*height + yBottom];
        bool neigh6 = input[x*height + yBottom];
        bool neigh7 = input[xRight*height + yBottom];
        int living_neighbors = neigh0 + neigh1 + neigh2 + neigh3 + neigh4 + neigh5 + neigh6 + neigh7;

        int cell = input[x*height + y];
        int index = cell ? 9 + living_neighbors : living_neighbors;

        output[x*height+y] = binary_rule[index];
    }
}


void cuda_updateCellularState(bool* input, bool* output, bool* binary_rule, int grid_size)
{
    dim3 block(32, 32, 1);
    dim3 grid((grid_size + block.x - 1)/block.x, (grid_size + block.y - 1)/block.y, 1);
    updateCellularState<<<grid, block>>>(input, output, binary_rule, grid_size);
}


