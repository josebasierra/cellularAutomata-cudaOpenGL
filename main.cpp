#include "ParticleSim.h"
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720


ParticleSim particleSim;
int elapsedtTime = 0;

void cleanup();
void drawCallback();
void idleCallback();


int main(int argc, char **argv)
{
    // init window
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow(argv[0]);
    
    // glut callable functions
    glutDisplayFunc(drawCallback);
    glutIdleFunc(idleCallback);
    glutCloseFunc(cleanup);

    // needed to use 'gl' calls
    glewInit();
    particleSim.init();
    
    glutMainLoop();
    
    return 0;
} 


void cleanup()
{
    //exit behaviour....
}


void drawCallback(){
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glm::mat4 modelview = glm::mat4(1.0f);
    glm::mat4 projection = glm::ortho(0.0f, float(SCREEN_WIDTH), float(SCREEN_HEIGHT), 0.0f);
    
    particleSim.draw(modelview, projection);
    
    glutSwapBuffers();
}


void idleCallback(){
    // update logic....
    
    
    int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);
	int deltaTime = newElapsedTime - elapsedtTime;
    elapsedtTime = newElapsedTime;
    
    particleSim.update(deltaTime);
    glutPostRedisplay();
}


