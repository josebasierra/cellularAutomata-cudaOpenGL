#include "CellularAutomata.h"
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

using namespace std::chrono;

#define SCREEN_WIDTH 900
#define SCREEN_HEIGHT 900
#define TIME_BETWEEN_SIMULATION_STEPS 50

CellularAutomata cellularAutomata;
glm::mat4 projection;

int elapsedTime = 0;
int timeSinceLastStep = 0;

int deltaTime = 0;
int drawTime = 0;
int updateTime = 0;

void cleanup();
void drawCallback();
void idleCallback();
void reshapeCallback(int width, int height);


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
    glutReshapeFunc(reshapeCallback);
    glutCloseFunc(cleanup);

    // needed to use 'gl' calls
    glewInit();
    
    //init camera
    projection = glm::ortho(-1.f, 1.f, -1.f , 1.f);
    
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


void updateTitle() 
{
    char title[256];
    sprintf(title, "Update: %i ms | Draw: %i ms | Total step time: %i ms", updateTime, drawTime, updateTime + drawTime);
    glutSetWindowTitle(title);
}


void cleanup()
{
    //exit behaviour....
}


void drawCallback()
{    
    int startTime = glutGet(GLUT_ELAPSED_TIME);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glm::mat4 modelview = glm::mat4(1.0f);
    
    //cellularAutomata.draw(modelview, projection);
 
    updateTitle();
    glutSwapBuffers();
    
    int endTime = glutGet(GLUT_ELAPSED_TIME);

    drawTime = endTime - startTime;
    cout << drawTime << endl;
}


void idleCallback()
{
    int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);
	deltaTime = newElapsedTime - elapsedTime;
    elapsedTime = newElapsedTime;
    
    if (elapsedTime - timeSinceLastStep >= TIME_BETWEEN_SIMULATION_STEPS){
        timeSinceLastStep = elapsedTime;
        

        auto start = chrono::steady_clock::now();
        int startTime = glutGet(GLUT_ELAPSED_TIME);
        cellularAutomata.update();
        int endTime = glutGet(GLUT_ELAPSED_TIME);
        auto end = chrono::steady_clock::now();
        
        updateTime = duration_cast<chrono::microseconds>(end - start).count();
        cout << updateTime << endl;
        glutPostRedisplay();
    }
}

//TODO: resize window correctly
void reshapeCallback(int width, int height){
    float ratio = width/(float)height;
    //projection = glm::ortho(-1.f, 1.f, -1.f*ratio , 1.f*ratio);
    cout << width << " " << height << endl;
}




