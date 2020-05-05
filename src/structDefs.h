#ifndef STRUCT_DEFS_H_
#define STRUCT_DEFS_H_


struct vec2 {
    float x, y;
};

struct rgba{
    unsigned char r,g,b,a;
};

struct Vertex {
    vec2 position;
    vec2 texCoord;
};

#endif
