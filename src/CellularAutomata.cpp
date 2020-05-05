#include "CellularAutomata.h"
#include "kernel.cuh"

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
using namespace std;

#define BUFFER_OFFSET(i) ((void*)(i))



CellularAutomata::~CellularAutomata() {
    
    free((void*)h_input);
    free((void*)h_output);
    cudaFree((void*)d_input);
    cudaFree((void*)d_output);
    
    freeQuadGeometry();
    freeTexture();
}


void CellularAutomata::init(){
    shaderProgram.loadShaders("src/shaders/texture.vert", "src/shaders/texture.frag");
    initQuadGeometry();
    initTexture();
    
    int numBytes = SIZE*SIZE*sizeof(bool);
    
    //host memory
    h_input = (bool*) malloc(numBytes);
    h_output = (bool*) malloc(numBytes);
    
    //device memory
    cudaMalloc((void**)&d_input, numBytes); 
    cudaMalloc((void**)&d_output, numBytes); 
    
    initState();
    simulation_step = 0;
}

//TODO: Adapt cuda code for CellularAutomata
void CellularAutomata::update() {
    string mode = "CUDA";
    
    if (mode == "CUDA") 
    {
        // change input/output
        if (simulation_step % 2) {
            bool* aux = d_input;
            d_input = d_output;
            d_output = aux;
        }
        cuda_updateCellularState(d_input, d_output, SIZE);
        
        //get data from gpu to cpu to update texture
        cudaMemcpy(h_output, d_output, SIZE*SIZE*sizeof(bool), cudaMemcpyDeviceToHost); 
    }
    else if (mode == "CPU")
    {
        if (simulation_step % 2) {
            bool* aux = h_input;
            h_input = h_output;
            h_output = aux;
        }
        updateCellularState(h_input, h_output);
    }
    else
    {
        cout << "fuck off" << endl;
    }
    simulation_step++;
    
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


void CellularAutomata::draw(glm::mat4& modelview, glm::mat4& projection) {
    
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

void CellularAutomata::initState(){
    for (int i = 0; i < SIZE*SIZE; i++){
        h_input[i] = (rand() % 100) < 50;
    }
      cudaMemcpy(d_input, h_input, SIZE*SIZE*sizeof(bool), cudaMemcpyHostToDevice);
}


void CellularAutomata::updateCellularState(bool* input, bool* output){
    for (int i = 0; i < SIZE*SIZE; i++)
    {
        output[i] = !input[i];
    }
}


void CellularAutomata::updateTexture(){
    rgba texture_data[SIZE*SIZE];
    
    for (int i = 0; i < SIZE*SIZE; i++){
        texture_data[i] = h_output[i] ? LIFE_COLOR : DEATH_COLOR;
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SIZE, SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data);
}


void CellularAutomata::initQuadGeometry() {
    
    // create VAO and VBO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    //TODO: change hardcoded positions
    //define quad vertices to display texture 
    Vertex v_left_up, v_left_down, v_right_up, v_right_down;
    v_left_up.position = {100,700};
    v_left_up.texCoord = {0,1};
    
    v_left_down.position = {100,100};
    v_left_down.texCoord = {0,0};
    
    v_right_up.position = {700, 700};
    v_right_up.texCoord = {1,1};
    
    v_right_down.position = {700,100};
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


void CellularAutomata::initTexture(){
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    int width = SIZE;
    int height = SIZE;
    
    rgba texture_data[width*height];
    for (int i = 0; i < width*height; i++){
        texture_data[i] = DEATH_COLOR;
    }
        
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data);
    glGenerateMipmap(GL_TEXTURE_2D);
}


void CellularAutomata::freeQuadGeometry(){
    // remove CUDA register of the VBO
    //cudaGraphicsUnregisterResource(cuda_vbo_resource);
    
    // delete openGL buffers
    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);
    vbo = 0;
}


void CellularAutomata::freeTexture(){
    //free texture memory...
}
