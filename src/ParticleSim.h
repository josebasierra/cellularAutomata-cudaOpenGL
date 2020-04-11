#ifndef _PARTICLE_SIM_INCLUDE
#define _PARTICLE_SIM_INCLUDE

#include "Particle.h"
#include "ShaderProgram.h"
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

#include <iostream>
using namespace std;


#define NUM_PARTICLES 100000
#define PARTICLE_SIZE 2
    

class ParticleSim {
    
public:
    ~ParticleSim();
    
    void init();
    void update(int deltaTime);
    void draw(glm::mat4& modelview, glm::mat4& projection);

private:
    void initParticles();
    void createVBO();
    void deleteVBO();
    
private:
    Particle h_particles[NUM_PARTICLES];
    Particle* d_particles;
    
    GLuint vao,vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;

    GLint posLocation;
    ShaderProgram shaderProgram;
    
    
}; 

#endif // _PARTICLE_SIM_INCLUDE 
