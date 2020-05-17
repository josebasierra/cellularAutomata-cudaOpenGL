#include "CellularAutomata.h"
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800
#define TIME_BETWEEN_SIMULATION_STEPS 50

CellularAutomata cellularAutomata;

int elapsedTime = 0;
int timeSinceLastStep = 0;
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
    
    //init cellularAutomata
    int rule = 6152;
    int grid_size = 300;
    string execution_mode = "cpu";
    
    if (argc > 1) rule = atoi(argv[1]);
    if (argc > 2) grid_size = atoi(argv[2]);
    if (argc > 3) execution_mode = argv[3];
    
    cellularAutomata.init(rule, grid_size, execution_mode);
    
    glutMainLoop();
    
    return 0;
} 


void updateAvgFps(int deltaTime) 
{
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


void drawCallback()
{    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glm::mat4 modelview = glm::mat4(1.0f);
    glm::mat4 projection = glm::ortho(0.0f, float(SCREEN_WIDTH), 0.0f , float(SCREEN_HEIGHT));
    
    cellularAutomata.draw(modelview, projection);
 
    glutSwapBuffers();
}


void idleCallback()
{
    // time shit...
    int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);
	int deltaTime = newElapsedTime - elapsedTime;
    elapsedTime = newElapsedTime;
    
    updateAvgFps(deltaTime);
    
    // update logic...
    if (elapsedTime - timeSinceLastStep >= TIME_BETWEEN_SIMULATION_STEPS){
        timeSinceLastStep = elapsedTime;
        
        cellularAutomata.update();
            
        glutPostRedisplay();
    }
}




