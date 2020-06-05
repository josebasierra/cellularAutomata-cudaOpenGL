#include "CellularAutomata.h"
#include "kernel.cuh"

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
using namespace std;

#define BUFFER_OFFSET(i) ((void*)(i))
#define MAX_STEPS 10000000

CellularAutomata::~CellularAutomata() 
{    
    free((void*)h_input);
    free((void*)h_output);
    cudaFree((void*)d_input);
    cudaFree((void*)d_output);
    
    freeQuadGeometry();
    freeTexture();
}


void CellularAutomata::init(int rule, int grid_size, string execution_mode)
{
    RuleToBinary(rule, h_binary_rule);
    this->grid_size = grid_size;
    this->execution_mode = execution_mode;
    
    int numBytes = grid_size*grid_size*sizeof(bool);
    
    //host memory
    h_input = (bool*) malloc(numBytes);
    h_output = (bool*) malloc(numBytes);
    
    //device memory
    cudaMalloc((void**)&d_input, numBytes); 
    cudaMalloc((void**)&d_output, numBytes); 
    
    cudaMalloc((void**)&d_binary_rule, RULE_SIZE*sizeof(bool));
    cudaMemcpy(d_binary_rule, h_binary_rule, RULE_SIZE*sizeof(bool), cudaMemcpyHostToDevice);

    
    initState();
    simulation_step = 0;
    
    //init drawing
    shaderProgram.loadShaders("src/shaders/texture.vert", "src/shaders/texture.frag");
    initQuadGeometry();
    initTexture();
}


void CellularAutomata::update() 
{    
    //if (simulation_step > MAX_STEPS) return;

    if (execution_mode == "cuda") 
    {
        if(simulation_step != 0)
        {
            bool* aux = d_input;
            d_input = d_output;
            d_output = aux;
        }
        cuda_updateCellularState(d_input, d_output, d_binary_rule, grid_size);

    }
    else if (execution_mode == "cpu")
    {    
        if(simulation_step != 0) 
        {
            bool* aux = h_input;
            h_input = h_output;
            h_output = aux;
        }
        updateCellularState(h_input, h_output, h_binary_rule);
    }

    simulation_step++;

    //TODO: cuda - opengl interop
    
    // update of VBO inside CUDA ------------------------------------------------------------
    
    // map vbo to be used by CUDA
    //Vertex *dptr;
    //cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    //size_t num_bytes;
    //cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,cuda_vbo_resource);
    
    // run kernel to update vertices of the VBO
    //run_updateVertices(dptr, d_particles, NUM_PARTICLES);
    
    // unmap vbo
    //cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    
    // --------------------------------------------------------------------------------------
}


void CellularAutomata::draw(glm::mat4& modelview, glm::mat4& projection) 
{    
    updateTexture();
    
    // activate program and pass uniforms to shaders
    shaderProgram.use();
	shaderProgram.setUniformMatrix4f("modelview", modelview);
    shaderProgram.setUniformMatrix4f("projection", projection);
    shaderProgram.setUniform4f("color", 1.0f, 0.0f, 0.0f, 1.0f);
    
    // bind
    glEnable(GL_TEXTURE_2D);
	glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
	glEnableVertexAttribArray(posLocation);
    glEnableVertexAttribArray(texLocation);
	glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // unbind
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_TEXTURE_2D);
}


// PRIVATE METHODS -----------------------------------------------------------------------


void CellularAutomata::RuleToBinary(int rule, bool *binary)    
{
    //in a 2D image, the neighbourhood has 9 cells
    for(int i = 0; i < RULE_SIZE; i++)    
    {    
        binary[i] = rule%2;    
        rule = rule/2;  
    }  
}


void CellularAutomata::initState()
{
    for (int i = 0; i < grid_size*grid_size; i++)
    {
        h_input[i] = (rand() % 100) < 50;
    }
    cudaMemcpy(d_input, h_input, grid_size*grid_size*sizeof(bool), cudaMemcpyHostToDevice);
}


void CellularAutomata::updateCellularState(bool* input, bool* output, bool* binary)
{
    int width = grid_size;
    int height = grid_size;

    for (int x = 0; x < width; x++) 
    {
        for (int y = 0; y < height; y++) 
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
            
            output[x*height+y] = binary[index];
        }
    }
}


void CellularAutomata::updateTexture()
{
    //get data from gpu to cpu to update texture
    if (execution_mode == "cuda") 
    {
        cudaMemcpy(h_output, d_output, grid_size*grid_size*sizeof(bool), cudaMemcpyDeviceToHost); 
    }
    
    rgba *texture_data = new rgba[grid_size*grid_size];
    for (int i = 0; i < grid_size*grid_size; i++)
    {
        texture_data[i] = h_output[i] ? LIFE_COLOR : DEATH_COLOR;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, grid_size, grid_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data);
    delete(texture_data);
}


void CellularAutomata::initQuadGeometry() 
{    
    // create VAO and VBO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    //TODO: change hardcoded positions
    //define quad vertices to display texture 
    Vertex v_left_up, v_left_down, v_right_up, v_right_down;
    v_left_up.position = {-1,1};
    v_left_up.texCoord = {0,1};
    
    v_left_down.position = {-1,-1};
    v_left_down.texCoord = {0,0};
    
    v_right_up.position = {1, 1};
    v_right_up.texCoord = {1,1};
    
    v_right_down.position = {1,-1};
    v_right_down.texCoord = {1.0};
    
    Vertex quad_vertices[6] = {
        v_left_down, v_left_up, v_right_up,
        v_left_down, v_right_up, v_right_down
    };
    
    glBufferData(GL_ARRAY_BUFFER, 6*sizeof(Vertex), quad_vertices, GL_DYNAMIC_DRAW);
    posLocation = shaderProgram.bindVertexAttribute("position", 2, sizeof(Vertex), 0);
    texLocation = shaderProgram.bindVertexAttribute("texCoord", 2, sizeof(Vertex), BUFFER_OFFSET(2*sizeof(float)) );
    
    // unbind
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    //register VBO with CUDA
    //unsigned int vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard;
    //cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, vbo_res_flags);
}


void CellularAutomata::initTexture()
{
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    int width = grid_size;
    int height = grid_size;
    
    rgba *texture_data = new rgba[width*height];
    for (int i = 0; i < width*height; i++){
        texture_data[i] = DEATH_COLOR;
    }
    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    delete(texture_data);
}


void CellularAutomata::freeQuadGeometry()
{
    // remove CUDA register of the VBO
    //cudaGraphicsUnregisterResource(cuda_vbo_resource);
    
    // delete openGL buffers
    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);
    vbo = 0;
}


void CellularAutomata::freeTexture()
{
    //free texture memory...
}
