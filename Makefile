OS=$(shell uname)
CC=gcc

FLAGS=-Wall -std=c99
PROFILER_FLAGS=-Wall -std=c99

ifeq ($(OS), Darwin)
	FLAGS+=-F/Library/Frameworks -framework SDL2 -framework OpenCL -framework OpenGL
	PROFILER_FLAGS+=-F/Library/Frameworks -framework OpenCL -framework OpenGL
else ifeq ($(OS), Linux)
	FLAGS+=-lSDL2 -lOpenCL -lGL -lm
	PROFILER_FLAGS+=-lOpenCL -lGL -lm
endif

C_FILES=main.c cl_fluid_sim.c sdl_window.c
H_FILES=cl_fluid_sim.h sdl_window.h
EXE=fluid

.PHONY: clean all

all: $(EXE) profiler

$(EXE): $(C_FILES) $(H_FILES)
	$(CC) $(C_FILES) $(FLAGS) -o$(EXE)

profiler: profiler.c cl_fluid_sim.c cl_fluid_sim.h
	$(CC) profiler.c cl_fluid_sim.c $(PROFILER_FLAGS) -oprofiler

clean:
	rm -f $(EXE) profiler
