/*
 * gcc profiler.c cl_fluid_sim.c -framework OpenCL -frameowork OpenGL -oprofiler
 */

#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <signal.h>

#include "cl_fluid_sim.h"

extern char * optarg;

FluidSim * my_fluid_sim;

volatile int is_running = 1;

void quit()
{
  is_running = 0;
}

int main(int argc, char ** argv)
{
  signal(SIGINT, quit);

  size_t sim_size = 512;
  int num_r_steps = 20;
  FLAGS flags = F_PROFILE;
  int has_chosen_type = 0;

  int ch;
  while ((ch = getopt(argc, argv, "n:t:r:")) != -1)
  {
    switch (ch)
    {
      case 'n':
        sim_size = atoi(optarg);
        break;
      case 't':
        if (strcmp(optarg, "CPU") == 0)
        {
          flags |= F_USE_CPU;
          has_chosen_type = 1;
        }
        else if (strcmp(optarg, "GPU") == 0)
        {
          flags |= F_USE_GPU;
          has_chosen_type = 1;
        }
        else {
          fprintf(stderr, "Invalid device type.\n");
          return 1;
        }
        break;
      case 'r':
        num_r_steps = atoi(optarg);
        break;
      default:
        break;
    }
  }

  if (!has_chosen_type)
  { //set defualt value
    flags |= F_USE_GPU;
  }

  // zero is never a valid texture
  my_fluid_sim = create_fluid_sim(0, "fluid_kernel.cl", sim_size, 0.00001f, 0.00001f, num_r_steps, flags);

#ifdef __APPLE__
  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

  double seconds = 0;

  while (is_running)
  {

#ifdef __APPLE__
    clock_gettime(CLOCK_REALTIME, &end);
    seconds = (double)((end.tv_sec + end.tv_nsec * 1e-9) - (double)(start.tv_sec + start.tv_nsec * 1e-9));
    clock_gettime(CLOCK_REALTIME, &start);
#endif

    enqueue_event(my_fluid_sim, 0.5, 0.5, 1, 1.f, IS_A_DENSITY);
    enqueue_event(my_fluid_sim, 0.5, 0.5, 1, 1.f, IS_B_DENSITY);
    enqueue_event(my_fluid_sim, 0.5, 0.5, 1, 1.f, IS_U_VELOCITY);
    enqueue_event(my_fluid_sim, 0.5, 0.5, 1, 1.f, IS_V_VELOCITY);

    simulate_next_frame(my_fluid_sim, seconds);
  }

  destroy_fluid_sim(my_fluid_sim);

  return 0;
}
