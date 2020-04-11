#include "kernel.cuh"
#include <math.h>


const float GRAVITY = -9.8;


__global__ void updateParticles(Particle *particles, int N, int deltaTime)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N) {
        particles[i].speed.y += GRAVITY*(deltaTime/1000.f);
        particles[i].position.y += particles[i].speed.y * (deltaTime/1000.f);
    }

}


__global__ void updateVertices(VertexParticle* vertices, Particle* particles, int N)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N) {
        vertices[i].position = particles[i].position;
    }
}


void run_simulation(Particle* particles, int N, int deltaTime) {
    dim3 block(32, 1, 1);
    dim3 grid(N/block.x, 1, 1);
    updateParticles<<<grid, block>>>(particles, N, deltaTime);
    
}


void run_updateVertices(VertexParticle* vertices, Particle* particles, int N) {
    dim3 block(32, 1, 1);
    dim3 grid(N/block.x, 1, 1);
    updateVertices<<<grid, block>>>(vertices, particles, N);
}


