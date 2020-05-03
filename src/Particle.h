#ifndef PARTICLE_H_
#define PARTICLE_H_


struct vec2 {
    float x, y;
};

struct rgba{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

struct Vertex {
    vec2 position;
    vec2 texCoord;
    //float r, g, b;
    
};


struct Particle {
    vec2 position;        
    vec2 speed;
    //float type;
    //float mass....    
}; 

#endif
