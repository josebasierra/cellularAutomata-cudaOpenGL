#ifndef _PARTICLE_SIM_INCLUDE
#define _PARTICLE_SIM_INCLUDE

#include "structDefs.h"
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
    
    void init(int rule);
    void update();
    void draw(glm::mat4& modelview, glm::mat4& projection);

private:

    void RuleToBinary(int rule, bool (&binary)[512]);   
    void initState();
    void initQuadGeometry();
    void initTexture();
    
    void updateCellularState(bool* input, bool* output, bool* binary);
    void updateTexture();
    
    void freeQuadGeometry();
    void freeTexture();
    
private:
    bool *h_input, *h_output;
    bool *d_input, *d_output, *d_rule;
    uint simulation_step;
    bool binary[512];

    GLuint vao,vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;

    GLint posLocation, texLocation;
    GLuint texture;
    ShaderProgram shaderProgram;
}; 

#endif // _PARTICLE_SIM_INCLUDE 
