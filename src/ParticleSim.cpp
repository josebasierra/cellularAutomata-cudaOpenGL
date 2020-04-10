#include "ParticleSim.h"
#include "kernel.cuh"

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
using namespace std;

#define BUFFER_OFFSET(i) ((void*)(i))



ParticleSim::~ParticleSim() {
    deleteVBO();
}


void ParticleSim::init(){
    
    //init particle positions
    for (int i = 0; i < NUM_PARTICLES; i++){
        particles[i].x = rand()%1300;
        particles[i].y = rand()%800;
    }
    
    shaderProgram.loadShaders("src/shaders/simple.vert", "src/shaders/simple.frag");
    createVBO();
}


void ParticleSim::update(int deltaTime) {
    // map vbo to be used by CUDA
    Particle *dptr;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,cuda_vbo_resource);
    
    // CUDA LOGIC...
    run_kernel(dptr, NUM_PARTICLES, deltaTime);
    
    // unmap vbo
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}


void ParticleSim::draw(glm::mat4& modelview, glm::mat4& projection) {
    
    // activate program and pass uniforms to shaders
    shaderProgram.use();
	shaderProgram.setUniformMatrix4f("modelview", modelview);
    shaderProgram.setUniformMatrix4f("projection", projection);
    shaderProgram.setUniform4f("color", 1.0f, 0.0f, 0.0f, 1.0f);
    
    // bind
	glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
	glEnableVertexAttribArray(posLocation);
    glPointSize(PARTICLE_SIZE);
	glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    
    // unbind
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void ParticleSim::createVBO() {
    
    // create VAO and VBO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES*sizeof(Particle), &particles, GL_DYNAMIC_DRAW);
    posLocation = shaderProgram.bindVertexAttribute("position", 2, sizeof(Particle), 0);
    
    // unbind
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    //register VBO with CUDA
    unsigned int vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, vbo_res_flags);
}


void ParticleSim::deleteVBO(){
    // remove CUDA register of the VBO
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    
    // delete openGL buffers
    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);
    vbo = 0;
}
