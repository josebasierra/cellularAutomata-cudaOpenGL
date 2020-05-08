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


void CellularAutomata::init(int rule){
    shaderProgram.loadShaders("src/shaders/texture.vert", "src/shaders/texture.frag");
    initQuadGeometry();
    initTexture();

    RuleToBinary(rule, binary);
    for (int i = 0; i < 512; i++)
        cout << binary[i] << " ";
    
    int numBytes = SIZE*SIZE*sizeof(bool);
    
    //host memory
    h_input = (bool*) malloc(numBytes);
    h_output = (bool*) malloc(numBytes);
    
    
    //device memory
    cudaMalloc((void**)&d_input, numBytes); 
    cudaMalloc((void**)&d_output, numBytes); 
    cudaMalloc((void**)&d_rule, 512*sizeof(bool));
    
    initState();
    simulation_step = 0;
}

//TODO: Adapt cuda code for CellularAutomata
void CellularAutomata::update() {
    string mode = "CPU";
    
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
        bool* aux = h_input;
        h_input = h_output;
        h_output = aux;
        updateCellularState(h_input, h_output, binary);
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


void CellularAutomata::RuleToBinary(int rule, bool (&binary)[512])    
{
    //in a 2D image, the neighbourhood has 9 cells
    for(int i = 0; i < 512; i++)    
    {    
        binary[i] = rule%2;    
        rule = rule/2;  
    }  
}


void CellularAutomata::initState(){
    for (int i = 0; i < SIZE*SIZE; i++){
        //h_input[i] = (rand() % 100) < 50;
        h_input[i] = false;
        h_output[i] = false;
    }
    h_input[SIZE*SIZE/2+ SIZE/2] = true;

    h_output[SIZE*SIZE/2+ SIZE/2] = true;
    cudaMemcpy(d_input, h_input, SIZE*SIZE*sizeof(bool), cudaMemcpyHostToDevice);
}

int toInt(bool b) {
    int a = 0;
    if (b) a = 1;
    return a;
}


void CellularAutomata::updateCellularState(bool* input, bool* output, bool* binary){

    int width = SIZE;
    int height = SIZE;

    for (int x = 0; x < width; x++){
        for (int y = 0; y < height; y++){
            if (x > 0 && y > 0 && x < width -1 && y < height -1){
                int pos1 = toInt(input[(x-1)*width + (y+1)]); 
                int pos2 = toInt(input[(x)*width + (y+1)]); 
                int pos3 = toInt(input[(x+1)*width + (y+1)]); 
                int pos4 = toInt(input[(x-1)*width + (y)]);
                int pos5 = toInt(input[(x)*width + (y)]);
                int pos6 = toInt(input[(x+1)*width + (y)]);
                int pos7 = toInt(input[(x-1)*width + (y-1)]);
                int pos8 = toInt(input[(x)*width + (y-1)]);
                int pos9 = toInt(input[(x+1)*width + (y-1)]);
                int index = pos1 + 2*pos2 + 4*pos3 + 8*pos4 + 16*pos5 + 32*pos6 + 64*pos7 + 128*pos8 + 256*pos9;
                //cout << pos1 << " " << pos2 << " "<< pos3 << " "<< pos4 << " "<< pos5 << " "<< pos6 << " "<< pos7 << " "<< pos8 << " "<< endl;
                //cout << index << endl;
                output[x*width+y] = binary[index];
                
                //if (output[x*width+y]) cout << "!" << x << " " << y  << " " << binary[index] << endl;
            }
            
        }
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
