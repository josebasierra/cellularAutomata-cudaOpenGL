#include "kernel.cuh"
#include <stdio.h>
#include <math.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>


#define SIZE 32

__global__ void updateCellularState_0(bool* input, bool* output, bool* binary_rule, int grid_size)
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

__global__ void updateCellularState_1(bool* input, bool* output, bool* binary_rule, int grid_size)
{
    __shared__ bool s_input[SIZE][SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int width = grid_size;
    int height = grid_size;
    
    s_input[tx][ty] = input[x*height + y]; 
    __syncthreads();

    
    if (x < width && y < height) 
    {
        if (threadIdx.x == 0 || threadIdx.x == SIZE-1 || threadIdx.y == 0 || threadIdx.y == SIZE-1) 
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

            int cell = s_input[tx][ty];
            int index = cell ? 9 + living_neighbors : living_neighbors;

            output[x*height+y] = binary_rule[index];
        }
        else
        {   
            int xRight = tx + 1;
            int xLeft = tx - 1 ;
            int yTop = ty + 1; 
            int yBottom = ty - 1;

            bool neigh0 = s_input[xLeft][yTop]; 
            bool neigh1 = s_input[tx][yTop]; 
            bool neigh2 = s_input[xRight][yTop]; 
            bool neigh3 = s_input[xLeft][ty];
            bool neigh4 = s_input[xRight][ty];
            bool neigh5 = s_input[xLeft][yBottom];
            bool neigh6 = s_input[tx][yBottom];
            bool neigh7 = s_input[xRight][yBottom];
            int living_neighbors = neigh0 + neigh1 + neigh2 + neigh3 + neigh4 + neigh5 + neigh6 + neigh7;

            bool cell = s_input[tx][ty];
            
            int index = cell ? 9 + living_neighbors : living_neighbors;

            output[x*height+y] = binary_rule[index];
        }
    }

}


void cuda_updateCellularState(bool* input, bool* output, bool* binary_rule, int grid_size)
{
    dim3 block(SIZE, SIZE, 1);
    dim3 grid((grid_size + block.x - 1)/block.x, (grid_size + block.y - 1)/block.y, 1);
    
    updateCellularState_0<<<grid, block>>>(input, output, binary_rule, grid_size);
    cudaDeviceSynchronize();
    
    updateCellularState_1<<<grid, block>>>(input, output, binary_rule, grid_size);
    cudaDeviceSynchronize();
}


__global__ void updateTexture(bool* state, cudaSurfaceObject_t surface, int grid_size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    int width = grid_size;
    int height = grid_size;

    if (x < width && y < height)
    {   
        uchar4 data = state[x*height + y] ? make_uchar4(255,255,255,255) : make_uchar4(40,40,40,255);
        surf2Dwrite(data, surface, x * sizeof(uchar4), y);
    }
}

void cuda_updateTexture(bool* state, cudaSurfaceObject_t surface, int grid_size) 
{
    dim3 block(SIZE, SIZE, 1);
    dim3 grid((grid_size + block.x - 1)/block.x, (grid_size + block.y - 1)/block.y, 1);
    
    updateTexture<<<grid,block>>>(state, surface, grid_size);
    cudaDeviceSynchronize();
}


