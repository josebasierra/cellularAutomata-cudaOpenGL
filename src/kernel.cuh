#ifndef KERNEL_H_
#define KERNEL_H_

#include "Particle.h"


void run_simulation(Particle* particles, int N, int deltaTime);
void run_updateVertices(VertexParticle* vertices, Particle* particles, int N); 

#endif
