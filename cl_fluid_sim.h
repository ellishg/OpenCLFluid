#ifndef __CL_FLUID_SIM
#define __CL_FLUID_SIM

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
  #include "OpenCL/opencl.h"
  #include "OpenGL/gl.h"
  #include "OpenGL/opengl.h"
#elif __linux__
  #include "CL/cl.h"
  #include "GL/gl.h"
#endif

#define KB 1024
#define MAX_KERNEL_FILE_SIZE (20 * KB)
#define MAX_DENSITY 1
#define RUN_BAD_DIFFUSE 0
#define MAX_NUM_SIMULTANEOUS_EVENTS 10
#define MAX_DT 0.05f

#define PREV 0
#define CUR 1

#define NUM_SAMPLES 10

#define check_error(err, str) check_for_error(err, str, __FILE__, __LINE__)

typedef enum FLAGS
{
  F_PROFILE = 0b0001,
  F_USE_CPU = 0b0010,
  F_USE_GPU = 0b0100
} FLAGS;

typedef enum VEC_TYPE
{
  IS_DENSITY,
  IS_A_DENSITY,
  IS_B_DENSITY,
  IS_VELOCITY,
  IS_U_VELOCITY,
  IS_V_VELOCITY,
  IS_NONE
} VEC_TYPE;

typedef struct source_event_list_t
{
  cl_int x[MAX_NUM_SIMULTANEOUS_EVENTS];
  cl_int y[MAX_NUM_SIMULTANEOUS_EVENTS];
  cl_float strength[MAX_NUM_SIMULTANEOUS_EVENTS];
  cl_int max_radius_sqrd[MAX_NUM_SIMULTANEOUS_EVENTS];
  cl_int num_events;
} SourceEventList;

typedef struct fluid_sim_t
{
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;

  cl_kernel add_event_sources_kernel;
  cl_kernel add_source_kernel;
  cl_kernel set_bnd_kernel;
  cl_kernel diffuse_bad_kernel;
  cl_kernel diffuse_kernel;
  cl_kernel advect_kernel;
  cl_kernel project_a_kernel;
  cl_kernel project_b_kernel;
  cl_kernel project_c_kernel;
  cl_kernel make_framebuffer_kernel;

  int profile;
  int is_using_opengl;

  cl_event add_event_sources_event;
  cl_event add_source_event;
  cl_event set_bnd_event;
  cl_event diffuse_bad_event;
  cl_event diffuse_event;
  cl_event advect_event;
  cl_event project_a_event;
  cl_event project_b_event;
  cl_event project_c_event;
  cl_event make_framebuffer_event;

  size_t calls_to_add_event_sources;
  size_t calls_to_add_source;
  size_t calls_to_set_bnd;
  size_t calls_to_diffuse_bad;
  size_t calls_to_diffuse;
  size_t calls_to_advect;
  size_t calls_to_project_a;
  size_t calls_to_project_b;
  size_t calls_to_project_c;
  size_t calls_to_make_framebuffer;

  size_t cur_sample;
  cl_ulong add_event_sources_samples[NUM_SAMPLES];
  cl_ulong add_source_samples[NUM_SAMPLES];
  cl_ulong set_bnd_samples[NUM_SAMPLES];
  cl_ulong diffuse_bad_samples[NUM_SAMPLES];
  cl_ulong diffuse_samples[NUM_SAMPLES];
  cl_ulong advect_samples[NUM_SAMPLES];
  cl_ulong project_a_samples[NUM_SAMPLES];
  cl_ulong project_b_samples[NUM_SAMPLES];
  cl_ulong project_c_samples[NUM_SAMPLES];
  cl_ulong make_framebuffer_samples[NUM_SAMPLES];

  cl_mem density_mem[2];
  cl_mem velocity_mem[2];
  cl_mem framebuffer;

  cl_mem source_x;
  cl_mem source_y;
  cl_mem source_strength;
  cl_mem source_max_radius_sqrd;

  size_t sim_size;
  size_t stride;

  size_t global_size[2];
  size_t local_size[2];
  size_t full_global_size;
  size_t full_local_size;
  size_t set_bnd_global_size;
  size_t set_bnd_local_size;

  SourceEventList a_density_events;
  SourceEventList b_density_events;
  SourceEventList u_velocity_events;
  SourceEventList v_velocity_events;

  float diffusion_rate;
  float viscosity;
} FluidSim;

FluidSim * create_fluid_sim(GLuint window_texture, const char * kernel_filename, size_t sim_size, float diff, float visc, FLAGS flags);

void destroy_fluid_sim(FluidSim * fluid);

void enqueue_event(FluidSim * fluid, float x, float y, float s, float max_r, VEC_TYPE vec_type);

void simulate_next_frame(FluidSim * fluid, float dt);

void copy_to_framebuffer(FluidSim * fluid, cl_mem * dest);

void density_step(FluidSim * fluid, float dt);

void velocity_step(FluidSim * fluid, float dt);

void diffuse(FluidSim * fluid, cl_mem * dest, cl_mem * src, cl_float a, VEC_TYPE vec_type);

void advect(FluidSim * fluid, cl_mem * dest, cl_mem * src, cl_mem * vel, cl_float dt, VEC_TYPE vec_type);

void project(FluidSim * fluid, cl_mem * vel, cl_mem * tmp);

void set_bnd(FluidSim * fluid, cl_mem * dest, VEC_TYPE vec_type);

void add_source(FluidSim * fluid, cl_mem * dest, cl_mem * src, cl_float dt);

void add_event_sources(FluidSim * fluid, cl_mem * dest, SourceEventList * events, cl_int vec_type);

void swap_dens_buffers(FluidSim * fluid);

void swap_vel_buffers(FluidSim * fluid);

float profile_event(cl_event event, size_t times_run, cl_ulong samples[NUM_SAMPLES], size_t cur_sample, size_t n, int entries, const char * str);

void check_for_error(cl_int err, const char * str, const char * file, int line_number);

#endif
