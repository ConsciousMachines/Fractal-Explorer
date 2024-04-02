

NVCC          := /usr/local/cuda/bin/nvcc -ccbin g++

ALL_CCFLAGS := -m64 # add -g -G for debug here
ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_CCFLAGS += --threads 0 --std=c++11

# Common includes and paths for CUDA
INCLUDES  := -I/home/chad/Desktop/_backups/notes/.ignore/cuda-samples-12.3/Common # cuda toolkit samples common
INCLUDES  += -Iusr/lib/x86_64-linux-gnu 
INCLUDES  += -I../Common/vendor -I../Common/vendor/imgui
LIBRARIES := -L/usr/lib64/nvidia 
LIBRARIES += -L../Common/lib
LIBRARIES += -lX11 -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lGL -lglfw3 # -lGLU

# Gencode arguments
SMS ?= 50 52 60 61 70 75 80 86 89 90
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := 90
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)






# Define OBJ Directory
OBJ_DIR := ../Common/obj

# # ImGui source files
IMGUI_DIR := ../Common/vendor/imgui
IMGUI_SOURCES := $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp 
IMGUI_SOURCES += $(IMGUI_DIR)/imgui_widgets.cpp $(IMGUI_DIR)/imgui_impl_glfw.cpp $(IMGUI_DIR)/imgui_impl_opengl3.cpp

# Convert .cpp files in IMGUI_SOURCES to .o files in OBJ_DIR
IMGUI_OBJECTS := $(patsubst $(IMGUI_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(IMGUI_SOURCES))














all: main

$(OBJ_DIR)/glad.o: ../Common/vendor/glad/glad.c
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# for ImGui 
$(OBJ_DIR)/%.o: $(IMGUI_DIR)/%.cpp | $(OBJ_DIR)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

kernel.o: kernel.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

camera.o: camera.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o: main.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main: main.o kernel.o camera.o ../Common/obj/glad.o $(IMGUI_OBJECTS)
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	rm -f *.o

run: main
	$(EXEC) primusrun ./main