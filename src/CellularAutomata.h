#ifndef _CELLULAR_AUTOMATA_INCLUDE
#define _CELLULAR_AUTOMATA_INCLUDE

#include "structDefs.h"
#include "ShaderProgram.h"
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

#include <string>
#include <iostream>
using namespace std;


const int RULE_SIZE = 18;
const rgba DEATH_COLOR = {40, 40, 40, 255};
const rgba LIFE_COLOR = {255, 255, 255, 255};

class CellularAutomata {
    
public:
    ~CellularAutomata();
    
    void init(int rule, int grid_size, string execution_mode);
    void update();
    void draw(glm::mat4& modelview, glm::mat4& projection);

private:
    void RuleToBinary(int rule, bool *binary_rule);   
    
    void initState();
    void initQuadGeometry();
    void initTexture();
    
    void updateCellularState(bool* input, bool* output, bool* binary_rule);
    void updateTexture();
    
    void freeQuadGeometry();
    void freeTexture();
    
private:
    int grid_size;
    string execution_mode;
    
    //host data
    bool *h_input, *h_output;
    bool h_binary_rule[RULE_SIZE];
    
    //device data
    bool *d_input, *d_output, *d_binary_rule;
    
    uint simulation_step;


    GLuint vao,vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;

    GLint posLocation, texLocation;
    GLuint texture;
    ShaderProgram shaderProgram;
}; 

#endif 
