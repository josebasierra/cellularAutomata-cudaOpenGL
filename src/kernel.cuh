#ifndef KERNEL_H_
#define KERNEL_H_

#include "structDefs.h"


void cuda_updateCellularState(bool* input, bool* output, bool* binary_rule, int grid_size);
//void run_update_texture(bool* output, direccionTexturaGPU...); 

#endif
