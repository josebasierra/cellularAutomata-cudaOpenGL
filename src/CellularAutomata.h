#ifndef _PARTICLE_SIM_INCLUDE
#define _PARTICLE_SIM_INCLUDE

#include "Particle.h"
#include "ShaderProgram.h"
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

#include <iostream>
using namespace std;

const int SIZE = 100;
const rgba DEATH_COLOR = {50, 50, 50, 255};
const rgba LIFE_COLOR = {255, 255, 255, 255};

class CellularAutomata {
    
public:
    ~CellularAutomata();
    
    void init();
    void update();
    void draw(glm::mat4& modelview, glm::mat4& projection);

private:
    void initState();
    void initVBO();
    void initTexture();
    
    void updateTexture();
    
    void deleteVBO();
    void deleteTexture();
    
private:
    bool state[SIZE*SIZE];
    uint simulation_step;
    
    GLuint vao,vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;

    GLint posLocation, texLocation;
    GLuint texture;
    ShaderProgram shaderProgram;
}; 

#endif // _PARTICLE_SIM_INCLUDE 
