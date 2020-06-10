#ifndef KERNEL_H_
#define KERNEL_H_

#include "structDefs.h"


void cuda_updateCellularState(bool* input, bool* output, bool* binary_rule, int grid_size);
void cuda_updateTexture(bool* state, cudaSurfaceObject_t surface, int grid_size); 

#endif
