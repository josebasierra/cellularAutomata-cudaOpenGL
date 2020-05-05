#ifndef KERNEL_H_
#define KERNEL_H_

#include "structDefs.h"


void cuda_updateCellularState(bool* input, bool* output, int SIZE);
//void run_update_texture(bool* output, direccionTexturaGPU...); 

#endif
