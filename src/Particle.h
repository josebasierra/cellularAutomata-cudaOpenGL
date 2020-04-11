#ifndef PARTICLE_H_
#define PARTICLE_H_


struct vec2 {
    float x, y;
};


struct VertexParticle {
    vec2 position;
    //float r, g, b;
    
};


struct Particle {
    vec2 position;        
    vec2 speed;
    //float type;
    //float mass....    
}; 

#endif
