cmake_minimum_required(VERSION 3.7)
project(OpenCLFluid)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(include ${SDL2_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS})

add_executable(fluid test/main.c src/cl_fluid_sim.c src/sdl_window.c)
target_link_libraries(fluid m ${SDL2_LIBRARIES} ${OPENGL_LIBRARIES} OpenCL::OpenCL)

add_executable(profiler test/profiler.c src/cl_fluid_sim.c src/sdl_window.c)
target_link_libraries(profiler m ${SDL2_LIBRARIES} ${OPENGL_LIBRARIES} OpenCL::OpenCL)
