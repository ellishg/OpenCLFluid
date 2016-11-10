OS=$(shell uname)
CC=gcc

FLAGS=-Wall -std=c99
PROFILER_FLAGS=-Wall -std=c99

ifeq ($(OS), Darwin)
	FLAGS+=-F/Library/Frameworks -framework SDL2 -framework OpenCL -framework OpenGL
	PROFILER_FLAGS+=-F/Library/Frameworks -framework OpenCL -framework OpenGL
else ifeq ($(OS), Linux)
	FLAGS+=-lSDL2 -lOpenCL -lGL
	PROFILER_FLAGS+=-lOpenCL -lGL
endif

C_FILES=main.c cl_fluid_sim.c sdl_window.c
H_FILES=cl_fluid_sim.h sdl_window.h
EXE=fluid

all: $(EXE) profiler

$(EXE): $(C_FILES) $(H_FILES)
	$(CC) $(FLAGS) $(C_FILES) -o$(EXE)

profiler: profiler.c cl_fluid_sim.c cl_fluid_sim.h
	$(CC) $(PROFILER_FLAGS) profiler.c cl_fluid_sim.c -oprofiler

clean:
	rm -f $(EXE) profiler
