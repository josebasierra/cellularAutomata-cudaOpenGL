#include "ParticleSim.h"
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

ParticleSim particleSim;

int elapsedtTime = 0;
float frames = 0;
float timeFrames = 0;


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
    glutCreateWindow("CUDA particle simulation");
    
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


void updateAvgFps(int deltaTime) {
    
    frames++;
    timeFrames+=deltaTime;
    
    // update fps display every second
    if (timeFrames >= 1000) {
        
        float avgfps = frames/(timeFrames/1000.0f);
        
        char title[256];
        sprintf(title, "CUDA Particle Simulation: %3.1f fps", avgfps);
        glutSetWindowTitle(title);
        
        frames = 0;
        timeFrames = 0;
    }
}


void cleanup()
{
    //exit behaviour....
}


void drawCallback(){
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glm::mat4 modelview = glm::mat4(1.0f);
    glm::mat4 projection = glm::ortho(0.0f, float(SCREEN_WIDTH), 0.0f , float(SCREEN_HEIGHT));
    
    particleSim.draw(modelview, projection);

    glutSwapBuffers();
}


void idleCallback(){

    // time shit...
    int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);
	int deltaTime = newElapsedTime - elapsedtTime;
    elapsedtTime = newElapsedTime;
    
    updateAvgFps(deltaTime);
    
    // update logic...
    particleSim.update(deltaTime);
    
    glutPostRedisplay();
}


