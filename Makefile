CUDA_HOME   	= /usr/local/cuda
NVCC        	= $(CUDA_HOME)/bin/nvcc
COMPILER	= $(NVCC)

NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

INCLUDES	= -I./ -I./utils
LIBRARY_FLAGS	= -lglut -lGL -lGLEW


OBJECTS := main.o ParticleSim.o kernel.o Shader.o ShaderProgram.o 


default: program.exe
	rm -rf *.o
	

program.exe: $(OBJECTS)
	$(COMPILER) -o $@ $(OBJECTS) $(LIBRARY_FLAGS) $(LD_FLAGS)  

	
%.o: %.cpp
	$(COMPILER) $(INCLUDES) -c $< -o $@
	
	
%.o: utils/%.cpp
	$(COMPILER) $(INCLUDES) -c $< -o $@
	

%.o: %.cu
	$(NVCC) $(INCLUDES) -c $< -o $@ $(NVCC_FLAGS)
	
	
clean:
	rm -rf *.o program*.exe

	
