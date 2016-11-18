#include "cl_fluid_sim.h"

cl_int err;

FluidSim * create_fluid_sim(GLuint window_texture, const char * kernel_filename, size_t sim_size, float diff, float visc, int num_r_steps, FLAGS flags)
{
  FluidSim * fluid = (FluidSim *)malloc(sizeof(FluidSim));

  fluid->profile = (flags & F_PROFILE) ? 1 : 0;

  // zero is always an invalid texture
  fluid->is_using_opengl = (window_texture) ? 1 : 0;

  // Read kernel file
  FILE * kernel_file = fopen(kernel_filename, "r");
  if (!kernel_file)
  {
    perror("Failed to open kernel file");
    return NULL;
  }

  char * kernel_src = (char *)malloc(MAX_KERNEL_FILE_SIZE);
  size_t kernel_src_size = fread(kernel_src, 1, MAX_KERNEL_FILE_SIZE, kernel_file);
  kernel_src[kernel_src_size] = '\0';

  fclose(kernel_file);

  fluid->sim_size = sim_size;
  fluid->stride = sim_size + 2;
  fluid->num_relaxation_steps = num_r_steps;
  fluid->diffusion_rate = diff;
  fluid->viscosity = visc;

  fluid->global_size[0] = fluid->sim_size;
  fluid->global_size[1] = fluid->sim_size;
  // TODO: be smarter about setting these values
  fluid->local_size[0] = 128;
  fluid->local_size[1] = 128;
  fluid->buffer_size = 2 * (fluid->sim_size + 2) * (fluid->sim_size + 2);
  fluid->full_local_size = 8;
  fluid->set_bnd_global_size = (fluid->sim_size + 1);
  fluid->set_bnd_local_size = 1;

  fluid->a_density_events.num_events = 0;
  fluid->b_density_events.num_events = 0;
  fluid->u_velocity_events.num_events = 0;
  fluid->v_velocity_events.num_events = 0;

  fluid->cur_sample = 0;

  cl_uint num_available_platforms = -1;
  cl_uint num_available_devices = -1;

  // Get Platform and Device Info
  err = clGetPlatformIDs(1, NULL, &num_available_platforms);
  check_error(err, "Unable to get the number of platform IDs");
  if (flags & F_DEBUG)
  {
    fprintf(stdout, "%d platform(s) available\n", num_available_platforms);
  }

  cl_platform_id platforms[num_available_platforms];

  err = clGetPlatformIDs(num_available_platforms, platforms, NULL);
  check_error(err, "Unable to get platform IDs");

  char name[100];

  for (int i = 0; i < num_available_platforms; i++)
  {
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 100, name, NULL);
    if (flags & F_DEBUG)
    {
      fprintf(stdout, "\tplatform[%i] name: %s\n", i, name);
    }
  }

  // TODO: Be smarter about picking a platform
  cl_platform_id fluid_platform = platforms[0];

  err = clGetDeviceIDs(fluid_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_available_devices);
  check_error(err, "Unable to get the number of device IDs");
  if (flags & F_DEBUG)
  {
    fprintf(stdout, "%d device(s) available\n", num_available_devices);
  }

  cl_device_id devices[num_available_devices];

  err = clGetDeviceIDs(fluid_platform, CL_DEVICE_TYPE_ALL, num_available_devices, devices, NULL);
  check_error(err, "Unable to get device IDs");

  cl_device_id fluid_device = NULL;
  cl_device_type device_type;

  // TODO: Be smarter about picking a device
  for (int i = 0; i < num_available_devices; i++)
  {
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 100, name, NULL);
    clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    if (flags & F_DEBUG)
    {
      fprintf(stdout, "\tdevice[%i] is %s\n", i, name);
    }
    switch (device_type)
    {
      case CL_DEVICE_TYPE_GPU:
        //fprintf(stdout, " (GPU)\n");
        if (flags & F_USE_GPU)
        {
          fluid_device = devices[i];
        }
        break;
      case CL_DEVICE_TYPE_CPU:
        //fprintf(stdout, " (CPU)\n");
        if (flags & F_USE_CPU)
        {
          fluid_device = devices[i];
        }
        break;
      default:
        //fprintf(stdout, " (?)\n");
        break;
    }
  }

  if (fluid_device == NULL)
  {
    check_error(-1, "Unable to find device");
  }

  if (flags & F_DEBUG)
  {
    clGetDeviceInfo(fluid_device, CL_DEVICE_NAME, 100, name, NULL);
    fprintf(stdout, "Chosen device: %s\n", name);
  }

  size_t max_work_item_dimensions;
  clGetDeviceInfo(fluid_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &max_work_item_dimensions, NULL);

  size_t max_work_item_size[max_work_item_dimensions];
  clGetDeviceInfo(fluid_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t[max_work_item_dimensions]), max_work_item_size, NULL);

  fluid->local_size[0] = fmin(fluid->sim_size, fmin(fluid->local_size[0], max_work_item_size[0]));
  fluid->local_size[1] = fmin(fluid->sim_size, fmin(fluid->local_size[1], fmax(1, max_work_item_size[1] / fluid->local_size[0])));
  if (flags & F_DEBUG)
  {
    fprintf(stdout, "Max work item size (%zu, %zu)\nlocal work size (%zu, %zu)\n", max_work_item_size[0], max_work_item_size[1], fluid->local_size[0], fluid->local_size[1]);
  }

  if (fluid->is_using_opengl)
  {
#ifdef __APPLE__
    CGLContextObj gl_context = CGLGetCurrentContext();
    CGLShareGroupObj gl_share_group = CGLGetShareGroup(gl_context);

    cl_context_properties properties[] = {
      CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)gl_share_group,
      0
    };
#elif __linux__
    cl_context_properties properties[] = {
      // TODO: put stuff here
      0
    };
#endif
    // Create OpenCL context
    fluid->context = clCreateContext(properties, 1, &fluid_device, NULL, NULL, &err);
    check_error(err, "Unable to create cl context");
  }
  else {
    fluid->context = clCreateContext(NULL, 1, &fluid_device, NULL, NULL, &err);
    check_error(err, "Unable to create cl context");
  }

  // commands are executed in-order
  fluid->command_queue = clCreateCommandQueue(fluid->context, fluid_device, CL_QUEUE_PROFILING_ENABLE, &err);
  check_error(err, "Unable to create command queue");

  fluid->program = clCreateProgramWithSource(fluid->context, 1, (const char **)&kernel_src, (const size_t *)&kernel_src_size, &err);
  free(kernel_src);
  check_error(err, "Unable to create program with source");

  char * kernel_definitions = malloc(256);
  snprintf(kernel_definitions, 256, "-D SIM_SIZE=%zu "
                                    "-D STRIDE=%zu "
                                    "-D DOUBLE_STRIDE=%zu "
                                    "-D MAX_DENSITY=%d "
                                    "-D IS_DENSITY=%d "
                                    "-D IS_A_DENSITY=%d "
                                    "-D IS_B_DENSITY=%d "
                                    "-D IS_VELOCITY=%d "
                                    "-D IS_U_VELOCITY=%d "
                                    "-D IS_V_VELOCITY=%d "
                                    , fluid->sim_size, fluid->stride, 2 * fluid->stride, MAX_DENSITY, IS_DENSITY, IS_A_DENSITY, IS_B_DENSITY, IS_VELOCITY, IS_U_VELOCITY, IS_V_VELOCITY);

  err = clBuildProgram(fluid->program, 1, &fluid_device, kernel_definitions, NULL, NULL);
  free(kernel_definitions);
  const size_t max_log_length = 16384;
  char log[max_log_length];
  clGetProgramBuildInfo(fluid->program, fluid_device, CL_PROGRAM_BUILD_LOG, max_log_length, log, NULL);
  fprintf(stderr, "%s", log);
  check_error(err, "Unable to build program");

  fluid->set_bnd_kernel = clCreateKernel(fluid->program, "set_bnd", &err);
  check_error(err, "Unable to create set_bnd");
  fluid->add_event_sources_kernel = clCreateKernel(fluid->program, "add_event_sources", &err);
  check_error(err, "Unable to create make_area");
  fluid->add_source_kernel = clCreateKernel(fluid->program, "add_source", &err);
  check_error(err, "Unable to create add_source");
  fluid->diffuse_bad_kernel = clCreateKernel(fluid->program, "diffuse_bad", &err);
  check_error(err, "Unable to create diffuse_bad");
  fluid->diffuse_kernel = clCreateKernel(fluid->program, "diffuse", &err);
  check_error(err, "Unable to create diffuse");
  fluid->advect_kernel = clCreateKernel(fluid->program, "advect", &err);
  check_error(err, "Unable to create advect");
  fluid->project_a_kernel = clCreateKernel(fluid->program, "project_A", &err);
  check_error(err, "Unable to create project_A");
  fluid->project_b_kernel = clCreateKernel(fluid->program, "project_B", &err);
  check_error(err, "Unable to create project_B");
  fluid->project_c_kernel = clCreateKernel(fluid->program, "project_C", &err);
  check_error(err, "Unable to create project_C");
  fluid->make_framebuffer_kernel = clCreateKernel(fluid->program, "make_framebuffer", &err);
  check_error(err, "Unable to create make_framebuffer");

  // TODO in the future, maybe I could optimize this to one memory location
  fluid->density_mem[0] = clCreateBuffer(fluid->context, CL_MEM_READ_WRITE, fluid->buffer_size * sizeof(cl_float), NULL, &err);
  check_error(err, "Unable to create buffer");
  fluid->density_mem[1] = clCreateBuffer(fluid->context, CL_MEM_READ_WRITE, fluid->buffer_size * sizeof(cl_float), NULL, &err);
  check_error(err, "Unable to create buffer");
  fluid->velocity_mem[0] = clCreateBuffer(fluid->context, CL_MEM_READ_WRITE, fluid->buffer_size * sizeof(cl_float), NULL, &err);
  check_error(err, "Unable to create buffer");
  fluid->velocity_mem[1] = clCreateBuffer(fluid->context, CL_MEM_READ_WRITE, fluid->buffer_size * sizeof(cl_float), NULL, &err);
  check_error(err, "Unable to create buffer");
  if (fluid->is_using_opengl)
  {
    fluid->framebuffer = clCreateFromGLTexture2D(fluid->context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, window_texture, &err);
    check_error(err, "Unable to create cl/gl texture");
  }
  fluid->source_x = clCreateBuffer(fluid->context, CL_MEM_READ_ONLY, MAX_NUM_SIMULTANEOUS_EVENTS * sizeof(cl_int), NULL, &err);
  check_error(err, "Unable to create buffer");
  fluid->source_y = clCreateBuffer(fluid->context, CL_MEM_READ_ONLY, MAX_NUM_SIMULTANEOUS_EVENTS * sizeof(cl_int), NULL, &err);
  check_error(err, "Unable to create buffer");
  fluid->source_strength = clCreateBuffer(fluid->context, CL_MEM_READ_ONLY, MAX_NUM_SIMULTANEOUS_EVENTS * sizeof(cl_float), NULL, &err);
  check_error(err, "Unable to create buffer");
  fluid->source_max_radius_sqrd = clCreateBuffer(fluid->context, CL_MEM_READ_ONLY, MAX_NUM_SIMULTANEOUS_EVENTS * sizeof(cl_int), NULL, &err);
  check_error(err, "Unable to create buffer");

  err = clFlush(fluid->command_queue);
  err |= clFinish(fluid->command_queue);
  check_error(err, "Unable to finish queue");

  return fluid;
}

void destroy_fluid_sim(FluidSim * fluid)
{
  clFlush(fluid->command_queue);
  clFinish(fluid->command_queue);

  clReleaseMemObject(fluid->density_mem[0]);
  clReleaseMemObject(fluid->density_mem[1]);
  clReleaseMemObject(fluid->velocity_mem[0]);
  clReleaseMemObject(fluid->velocity_mem[1]);
  clReleaseMemObject(fluid->framebuffer);
  clReleaseMemObject(fluid->source_x);
  clReleaseMemObject(fluid->source_y);
  clReleaseMemObject(fluid->source_strength);
  clReleaseMemObject(fluid->source_max_radius_sqrd);

  clReleaseKernel(fluid->set_bnd_kernel);
  clReleaseKernel(fluid->add_event_sources_kernel);
  clReleaseKernel(fluid->add_source_kernel);
  clReleaseKernel(fluid->diffuse_bad_kernel);
  clReleaseKernel(fluid->diffuse_kernel);
  clReleaseKernel(fluid->advect_kernel);
  clReleaseKernel(fluid->project_a_kernel);
  clReleaseKernel(fluid->project_b_kernel);
  clReleaseKernel(fluid->project_c_kernel);
  if (fluid->is_using_opengl)
  {
    clReleaseKernel(fluid->make_framebuffer_kernel);
  }

  clReleaseProgram(fluid->program);
  clReleaseCommandQueue(fluid->command_queue);
  clReleaseContext(fluid->context);

  free(fluid);
}

void simulate_next_frame(FluidSim * fluid, float dt)
{
  fluid->calls_to_add_event_sources = 0;
  fluid->calls_to_add_source = 0;
  fluid->calls_to_set_bnd = 0;
  fluid->calls_to_diffuse_bad = 0;
  fluid->calls_to_diffuse = 0;
  fluid->calls_to_advect = 0;
  fluid->calls_to_project_a = 0;
  fluid->calls_to_project_b = 0;
  fluid->calls_to_project_c = 0;
  fluid->calls_to_make_framebuffer = 0;

  cl_float pattern = 0;
  err = clEnqueueFillBuffer(fluid->command_queue, fluid->density_mem[PREV], (void *)&pattern, sizeof(cl_float), 0, fluid->buffer_size * sizeof(cl_float), 0, NULL, NULL);
  err |= clEnqueueFillBuffer(fluid->command_queue, fluid->velocity_mem[PREV], (void *)&pattern, sizeof(cl_float), 0, fluid->buffer_size * sizeof(cl_float), 0, NULL, NULL);
  check_error(err, "Unable to clear buffers");

  add_event_sources(fluid, &fluid->density_mem[PREV], &fluid->a_density_events, IS_A_DENSITY);
  add_event_sources(fluid, &fluid->density_mem[PREV], &fluid->b_density_events, IS_B_DENSITY);
  add_event_sources(fluid, &fluid->velocity_mem[PREV], &fluid->u_velocity_events, IS_U_VELOCITY);
  add_event_sources(fluid, &fluid->velocity_mem[PREV], &fluid->v_velocity_events, IS_V_VELOCITY);

  float sim_dt = fmin(dt, MAX_DT);
  velocity_step(fluid, sim_dt);
  density_step(fluid, sim_dt);

  copy_to_framebuffer(fluid, &fluid->density_mem[CUR]);

  err = clFlush(fluid->command_queue);
  err |= clFinish(fluid->command_queue);
  check_error(err, "Unable to finish queue");

  if (fluid->profile)
  {
    float total_ms = profile_event(fluid->add_event_sources_event, fluid->calls_to_add_event_sources, fluid->add_event_sources_samples, fluid->cur_sample, fluid->sim_size, 1, "add_event_sources");
    total_ms += profile_event(fluid->add_source_event, fluid->calls_to_add_source, fluid->add_event_sources_samples, fluid->cur_sample, fluid->sim_size, 2, "add_source");
    total_ms += profile_event(fluid->set_bnd_event, fluid->calls_to_set_bnd, fluid->set_bnd_samples, fluid->cur_sample, 1, fluid->sim_size * 8, "set_bnd");
    if (RUN_BAD_DIFFUSE)
    {
      total_ms += profile_event(fluid->diffuse_bad_event, fluid->calls_to_diffuse_bad, fluid->diffuse_bad_samples, fluid->cur_sample, fluid->sim_size, 10, "diffuse_bad");
    }
    else {
      total_ms += profile_event(fluid->diffuse_event, fluid->calls_to_diffuse, fluid->diffuse_samples, fluid->cur_sample, fluid->sim_size, 10, "diffuse");
    }
    total_ms += profile_event(fluid->advect_event, fluid->calls_to_advect, fluid->advect_samples, fluid->cur_sample, fluid->sim_size, 10, "advect");
    total_ms += profile_event(fluid->project_a_event, fluid->calls_to_project_a, fluid->project_a_samples, fluid->cur_sample, fluid->sim_size, 4, "project_a");
    total_ms += profile_event(fluid->project_b_event, fluid->calls_to_project_b, fluid->project_b_samples, fluid->cur_sample, fluid->sim_size, 5, "project_b");
    total_ms += profile_event(fluid->project_c_event, fluid->calls_to_project_c, fluid->project_c_samples, fluid->cur_sample, fluid->sim_size, 6, "project_c");
    if (fluid->is_using_opengl)
    {
      total_ms += profile_event(fluid->make_framebuffer_event, fluid->calls_to_make_framebuffer, fluid->make_framebuffer_samples, fluid->cur_sample, fluid->sim_size, 2, "make_framebuffer");
    }

    fprintf(stdout, "Total GPU runtime: %.3f ms\nTotal wallclock time: %.0f ms\n\n", total_ms, 1000.f * dt);
    //fprintf(stdout, "%.3f\n%0.f\n\n", total_ms, 1000.f * dt);
    fluid->cur_sample = (fluid->cur_sample + 1) % NUM_SAMPLES;
  }
}

void velocity_step(FluidSim * fluid, float dt)
{
  add_source(fluid, &fluid->velocity_mem[CUR], &fluid->velocity_mem[PREV], dt);

  swap_vel_buffers(fluid);

  diffuse(fluid, &fluid->velocity_mem[CUR], &fluid->velocity_mem[PREV], dt * fluid->viscosity * fluid->sim_size * fluid->sim_size, IS_VELOCITY);

  project(fluid, &fluid->velocity_mem[CUR], &fluid->velocity_mem[PREV]);

  swap_vel_buffers(fluid);

  advect(fluid, &fluid->velocity_mem[CUR], &fluid->velocity_mem[PREV], &fluid->velocity_mem[PREV], dt, IS_VELOCITY);

  project(fluid, &fluid->velocity_mem[CUR], &fluid->velocity_mem[PREV]);
}

void density_step(FluidSim * fluid, float dt)
{
  add_source(fluid, &fluid->density_mem[CUR], &fluid->density_mem[PREV], dt);

  swap_dens_buffers(fluid);

  diffuse(fluid, &fluid->density_mem[CUR], &fluid->density_mem[PREV], dt * fluid->diffusion_rate * fluid->sim_size * fluid->sim_size, IS_DENSITY);

  swap_dens_buffers(fluid);

  advect(fluid, &fluid->density_mem[CUR], &fluid->density_mem[PREV], &fluid->velocity_mem[CUR], dt, IS_DENSITY);
}

void diffuse(FluidSim * fluid, cl_mem * dest, cl_mem * src, cl_float a, VEC_TYPE vec_type)
{
  if (RUN_BAD_DIFFUSE)
  {
    //__kernel void diffuse_bad(__global float * dest, __global float * src, float a)
    err = clSetKernelArg(fluid->diffuse_bad_kernel, 0, sizeof(cl_mem), dest);
    err |= clSetKernelArg(fluid->diffuse_bad_kernel, 1, sizeof(cl_mem), src);
    err |= clSetKernelArg(fluid->diffuse_bad_kernel, 2, sizeof(cl_float), &a);
    check_error(err, "Unable to set args");

    // enqueue diffuse_bad
    err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->diffuse_bad_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->diffuse_bad_event);
    check_error(err, "Unable to enqueue kernel");
    fluid->calls_to_diffuse_bad++;
  }
  else {

    cl_float denominator = 1 / (1 + 4 * a);

    for (int k = 0; k < fluid->num_relaxation_steps; k++)
    {
      //__kernel void diffuse(__global float * dest, __global float * src, float a, float denominator)
      err = clSetKernelArg(fluid->diffuse_kernel, 0, sizeof(cl_mem), dest);
      err |= clSetKernelArg(fluid->diffuse_kernel, 1, sizeof(cl_mem), src);
      err |= clSetKernelArg(fluid->diffuse_kernel, 2, sizeof(cl_float), &a);
      err |= clSetKernelArg(fluid->diffuse_kernel, 3, sizeof(cl_float), &denominator);
      check_error(err, "Unable to set args");

      // enqueue diffuse
      err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->diffuse_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->diffuse_event);
      check_error(err, "Unable to enqueue kernel");
      fluid->calls_to_diffuse++;

      set_bnd(fluid, dest, vec_type);
    }
  }
}

void advect(FluidSim * fluid, cl_mem * dest, cl_mem * src, cl_mem * vel, cl_float dt, VEC_TYPE vec_type)
{

  dt = -dt * fluid->sim_size;

  //__kernel void advect(__global float * dest, __global float * src, __global float * vel, float dt)
  err = clSetKernelArg(fluid->advect_kernel, 0, sizeof(cl_mem), dest);
  err |= clSetKernelArg(fluid->advect_kernel, 1, sizeof(cl_mem), src);
  err |= clSetKernelArg(fluid->advect_kernel, 2, sizeof(cl_mem), vel);
  err |= clSetKernelArg(fluid->advect_kernel, 3, sizeof(cl_float), &dt);
  check_error(err, "Unable to set args");

  // enqueue advect
  err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->advect_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->advect_event);
  check_error(err, "Unable to enqueue kernel");
  fluid->calls_to_advect++;

  set_bnd(fluid, dest, vec_type);
}

void project(FluidSim * fluid, cl_mem * vel, cl_mem * tmp)
{
  cl_float h = 0.5f / fluid->sim_size;

  //__kernel void project_A(__global float * tmp, __global float * vel, float h)
  err = clSetKernelArg(fluid->project_a_kernel, 0, sizeof(cl_mem), tmp);
  err |= clSetKernelArg(fluid->project_a_kernel, 1, sizeof(cl_mem), vel);
  err |= clSetKernelArg(fluid->project_a_kernel, 2, sizeof(cl_float), &h);
  check_error(err, "Unable to set args");

  // enqueue project_a
  err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->project_a_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->project_a_event);
  check_error(err, "Unable to enqueue kernel");
  fluid->calls_to_project_a++;

  set_bnd(fluid, tmp, IS_NONE);

  for (int k = 0; k < fluid->num_relaxation_steps; k++)
  {
    //__kernel void project_B(__global float * tmp)
    err = clSetKernelArg(fluid->project_b_kernel, 0, sizeof(cl_mem), tmp);
    check_error(err, "Unable to set args");

    // enqueue project_b
    err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->project_b_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->project_b_event);
    check_error(err, "Unable to enqueue kernel");
    fluid->calls_to_project_b++;

    set_bnd(fluid, tmp, IS_NONE);
  }

  h = 0.5f * fluid->sim_size;

  //__kernel void project_C(__global float * vel, __global float * tmp, float h)
  err = clSetKernelArg(fluid->project_c_kernel, 0, sizeof(cl_mem), vel);
  err |= clSetKernelArg(fluid->project_c_kernel, 1, sizeof(cl_mem), tmp);
  err |= clSetKernelArg(fluid->project_c_kernel, 2, sizeof(cl_float), &h);
  check_error(err, "Unable to set args");

  // enqueue project_c
  err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->project_c_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->project_c_event);
  check_error(err, "Unable to enqueue kernel");
  fluid->calls_to_project_c++;

  set_bnd(fluid, vel, IS_VELOCITY);
}

void add_source(FluidSim * fluid, cl_mem * dest, cl_mem * src, cl_float dt)
{
  //__kernel void add_source(__global float * dest, __global float * src, float dt)
  err = clSetKernelArg(fluid->add_source_kernel, 0, sizeof(cl_mem), dest);
  err |= clSetKernelArg(fluid->add_source_kernel, 1, sizeof(cl_mem), src);
  err |= clSetKernelArg(fluid->add_source_kernel, 2, sizeof(cl_float), &dt);
  check_error(err, "Unable to set args");

  // enqueue add_source
  err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->add_source_kernel, 1, NULL, &fluid->buffer_size, &fluid->full_local_size, 0, NULL, &fluid->add_source_event);
  check_error(err, "Unable to enqueue add_source");
  fluid->calls_to_add_source++;
}

void set_bnd(FluidSim * fluid, cl_mem * dest, VEC_TYPE vec_type)
{
  //__kernel void set_bnd(__global float * dest, int vec_type)
  err = clSetKernelArg(fluid->set_bnd_kernel, 0, sizeof(cl_mem), dest);
  err |= clSetKernelArg(fluid->set_bnd_kernel, 1, sizeof(cl_int), &vec_type);
  check_error(err, "Unable to set args");

  // enqueue set_bnd
  err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->set_bnd_kernel, 1, NULL, &fluid->set_bnd_global_size, &fluid->set_bnd_local_size, 0, NULL, &fluid->set_bnd_event);
  check_error(err, "Unable to enqueue set_bnd");
  fluid->calls_to_set_bnd++;
}

void copy_to_framebuffer(FluidSim * fluid, cl_mem * src)
{
  if (fluid->is_using_opengl)
  {
    glFinish();

    err = clEnqueueAcquireGLObjects(fluid->command_queue, 1, &fluid->framebuffer, 0, 0, NULL);
    check_error(err, "Unable to acquire texture");

    //__kernel void make_framebuffer(write_only image2d_t dest, __global float * src)
    err = clSetKernelArg(fluid->make_framebuffer_kernel, 0, sizeof(cl_mem), &fluid->framebuffer);
    err |= clSetKernelArg(fluid->make_framebuffer_kernel, 1, sizeof(cl_mem), src);
    check_error(err, "Unable to set args");

    //enqueue make_framebuffer
    err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->make_framebuffer_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->make_framebuffer_event);
    check_error(err, "Unable to enque make_framebuffer");
    fluid->calls_to_make_framebuffer++;

    err = clFlush(fluid->command_queue);
    err |= clFinish(fluid->command_queue);
    check_error(err, "Unable to finish queue");

    err = clEnqueueReleaseGLObjects(fluid->command_queue, 1, &fluid->framebuffer, 0, 0, NULL);
    check_error(err, "Unable to release texture");
  }
}

void enqueue_event(FluidSim * fluid, float x, float y, float s, float max_r, VEC_TYPE vec_type)
{
  SourceEventList * source_event = NULL;

  switch (vec_type) {
    case IS_A_DENSITY:
      source_event = &fluid->a_density_events;
      break;
    case IS_B_DENSITY:
      source_event = &fluid->b_density_events;
      break;
    case IS_U_VELOCITY:
      source_event = &fluid->u_velocity_events;
      break;
    case IS_V_VELOCITY:
      source_event = &fluid->v_velocity_events;
      break;
    default:
      check_error(1, "Invalid vec type");
      break;
  }

  if (source_event->num_events < MAX_NUM_SIMULTANEOUS_EVENTS)
  {
    source_event->x[source_event->num_events] = x * fluid->sim_size;
    source_event->y[source_event->num_events] = y * fluid->sim_size;
    source_event->strength[source_event->num_events] = s;
    source_event->max_radius_sqrd[source_event->num_events++] = max_r * max_r * fluid->sim_size * fluid->sim_size;
  }
  else {check_error(source_event->num_events, "Too many events");}
}

void add_event_sources(FluidSim * fluid, cl_mem * dest, SourceEventList * events, cl_int vec_type)
{
  if (events->num_events > 0)
  {
    err = clEnqueueWriteBuffer(fluid->command_queue, fluid->source_x, CL_TRUE, 0, events->num_events * sizeof(cl_int), events->x, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(fluid->command_queue, fluid->source_y, CL_TRUE, 0, events->num_events * sizeof(cl_int), events->y, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(fluid->command_queue, fluid->source_strength, CL_TRUE, 0, events->num_events * sizeof(cl_float), events->strength, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(fluid->command_queue, fluid->source_max_radius_sqrd, CL_TRUE, 0, events->num_events * sizeof(cl_int), events->max_radius_sqrd, 0, NULL, NULL);
    check_error(err, "Unable to write to buffer");

    //__kernel void add_event_sources(__global float * dest, __constant int * x, __constant int * y, __constant float * strength, __constant int * max_radius_sqrd, int num_events, int vec_type)
    err = clSetKernelArg(fluid->add_event_sources_kernel, 0, sizeof(cl_mem), dest);
    err |= clSetKernelArg(fluid->add_event_sources_kernel, 1, sizeof(cl_mem), &fluid->source_x);
    err |= clSetKernelArg(fluid->add_event_sources_kernel, 2, sizeof(cl_mem), &fluid->source_y);
    err |= clSetKernelArg(fluid->add_event_sources_kernel, 3, sizeof(cl_mem), &fluid->source_strength);
    err |= clSetKernelArg(fluid->add_event_sources_kernel, 4, sizeof(cl_mem), &fluid->source_max_radius_sqrd);
    err |= clSetKernelArg(fluid->add_event_sources_kernel, 5, sizeof(cl_int), &events->num_events);
    err |= clSetKernelArg(fluid->add_event_sources_kernel, 6, sizeof(cl_int), &vec_type);
    check_error(err, "Unable to set add_event_sources args");

    err = clEnqueueNDRangeKernel(fluid->command_queue, fluid->add_event_sources_kernel, 2, NULL, fluid->global_size, fluid->local_size, 0, NULL, &fluid->add_event_sources_event);
    check_error(err, "Unable to enqueue add_event_sources");
    fluid->calls_to_add_event_sources++;
  }

  events->num_events = 0;
}

void swap_dens_buffers(FluidSim * fluid)
{
  cl_mem tmp = fluid->density_mem[CUR];
  fluid->density_mem[CUR] = fluid->density_mem[PREV];
  fluid->density_mem[PREV] = tmp;
}

void swap_vel_buffers(FluidSim * fluid)
{
  cl_mem tmp = fluid->velocity_mem[CUR];
  fluid->velocity_mem[CUR] = fluid->velocity_mem[PREV];
  fluid->velocity_mem[PREV] = tmp;
}

float profile_event(cl_event event, size_t times_run, cl_ulong samples[NUM_SAMPLES], size_t cur_sample, size_t n, int entries, const char * str)
{
  cl_ulong time_start, time_end;

  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  check_error(err, "Unable to get profiling info");

  samples[cur_sample] = time_end - time_start;

  cl_ulong sum = 0;
  for (size_t i = 0; i < NUM_SAMPLES; i++)
  {
    sum += samples[i];
  }

  float ms = sum / NUM_SAMPLES / 1000000.f;

  float bandwidth = n * n * 4 * entries / ms * 1000 / (1024 * 1024 * 1024);

  fprintf(stdout, "%.3f ms at %.2f GB/s (ran %lu times) for %s\n", ms * times_run, bandwidth, times_run, str);
  //fprintf(stdout, "%.3f\n", ms * times_run);

  return ms * times_run;
}

void check_for_error(cl_int err, const char * str, const char * file, int line_number)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "<%s>:%d %s: %d\n", file, line_number, str, (int)err);
    exit(0);
  }
}
