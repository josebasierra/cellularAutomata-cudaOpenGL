#include "CellularAutomata.h"
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>


#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080
#define TIME_BETWEEN_SIMULATION_STEPS 50


CellularAutomata cellularAutomata;
int gridSize;
glm::mat4 modelview, projection;
glm::vec3 moveDirection;

int elapsedTime = 0;
int timeSinceLastStep = 0;

int deltaTime = 0;
int drawTime = 0;
int updateTime = 0;


void drawCallback();
void idleCallback();
void reshapeCallback(int width, int height);
void keyboardDownCallback(unsigned char key, int x, int y);
void keyboardUpCallback(unsigned char key, int x, int y);
void cleanup();


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
    glutKeyboardFunc(keyboardDownCallback);
    glutKeyboardUpFunc(keyboardUpCallback);
    glutCloseFunc(cleanup);

    // needed to use 'gl' calls
    glewInit();
    
    //init cellularAutomata
    int rule = 6152;
    gridSize = 600;
    string execution_mode = "cpu";
    
    if (argc > 1) rule = atoi(argv[1]);
    if (argc > 2) gridSize = atoi(argv[2]);
    if (argc > 3) execution_mode = argv[3];
    
    cellularAutomata.init(rule, gridSize, execution_mode);
    
    //init camera
    modelview = glm::translate(glm::mat4(1.0f), glm::vec3(0,0,-400 * 1.f/gridSize));        
    projection = glm::perspective(1.f,SCREEN_WIDTH/(float)SCREEN_HEIGHT,0.0f,10.0f);
    moveDirection = glm::vec3(0,0,0);
        
    glutMainLoop();
    return 0;
} 


void updateTitle() 
{
    char title[256];
    sprintf(title, "Size: %i x %i | Update: %.3f ms | Draw: %.3f ms | Total step time: %.3f ms", gridSize, gridSize, updateTime/1000.f, drawTime/1000.f, (updateTime + drawTime)/1000.f);
    glutSetWindowTitle(title);
}


void drawCallback()
{    
    auto start = chrono::steady_clock::now();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    cellularAutomata.draw(modelview, projection);
 
    updateTitle();
    glutSwapBuffers();
    
    auto end = chrono::steady_clock::now();
    drawTime = chrono::duration_cast<chrono::microseconds>(end - start).count();
}


void idleCallback()
{
    int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);
	deltaTime = newElapsedTime - elapsedTime;
    elapsedTime = newElapsedTime;
    
    //move camera
    float speed = 1000.0f/gridSize;
    auto translation = moveDirection * glm::vec3(speed * deltaTime/1000.f);
    modelview = glm::translate(modelview, translation);        

    
    if (elapsedTime - timeSinceLastStep >= TIME_BETWEEN_SIMULATION_STEPS){
        timeSinceLastStep = elapsedTime;
        
        auto start = chrono::steady_clock::now();
        cellularAutomata.update();
        auto end = chrono::steady_clock::now();
        updateTime = chrono::duration_cast<chrono::microseconds>(end - start).count();

        glutPostRedisplay();
    }
}


void reshapeCallback(int width, int height)
{
    glViewport(0, 0, width, height);
    projection = glm::perspective(1.f,width/(float)height,0.0f,10.0f);
}


void keyboardDownCallback(unsigned char key, int x, int y) 
{
    if (key == 'W' || key == 'w') moveDirection.y = -1;
    if (key == 'A' || key == 'a') moveDirection.x = 1;
    if (key == 'S' || key == 's') moveDirection.y = 1;
    if (key == 'D' || key == 'd') moveDirection.x = -1;
    
    if (key == '+') moveDirection.z = 1;
    if (key == '-') moveDirection.z = -1;
}


void keyboardUpCallback(unsigned char key, int x, int y) 
{
    float speed = 1.0f;
    if (key == 'W' || key == 'w') moveDirection.y = 0;
    if (key == 'A' || key == 'a') moveDirection.x = 0;
    if (key == 'S' || key == 's') moveDirection.y = 0;
    if (key == 'D' || key == 'd') moveDirection.x = 0;
    
    if (key == '+') moveDirection.z = 0;
    if (key == '-') moveDirection.z = 0;
}


void cleanup()
{
    //exit behaviour....
}




