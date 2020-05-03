#include "CellularAutomata.h"
#include "kernel.cuh"

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
using namespace std;

#define BUFFER_OFFSET(i) ((void*)(i))



CellularAutomata::~CellularAutomata() {
    deleteVBO();
}


void CellularAutomata::init(){
    shaderProgram.loadShaders("src/shaders/texture.vert", "src/shaders/texture.frag");
    initVBO();
    initTexture();
    
    initState();
    simulation_step = 0;
}

//TODO: Adapt cuda code for CellularAutomata
void CellularAutomata::update() {

    // ejemplo version cpu (faltaria otro state, input output)
    for (int i = 0; i < SIZE*SIZE; i++)
    {
        if (simulation_step % 2 && i%2){
            state[i] = !state[i];
        }
    }
    simulation_step++;
    updateTexture();
    
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


void CellularAutomata::initState(){
    for (int i = 0; i < SIZE*SIZE; i++){
        state[i] = false;
    }
}


void CellularAutomata::updateTexture(){
    rgba texture_data[SIZE*SIZE];
    for (int i = 0; i < SIZE*SIZE; i++){
        texture_data[i] = state[i] ? LIFE_COLOR : DEATH_COLOR;
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SIZE, SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data);
}


void CellularAutomata::initVBO() {
    
    // create VAO and VBO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

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


void CellularAutomata::deleteVBO(){
    // remove CUDA register of the VBO
    //cudaGraphicsUnregisterResource(cuda_vbo_resource);
    
    // delete openGL buffers
    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);
    vbo = 0;
}
