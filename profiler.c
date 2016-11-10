/*
 * gcc profiler.c cl_fluid_sim.c -framework OpenCL -frameowork OpenGL -oprofiler
 */

#include <unistd.h>
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
  FLAGS flags = F_PROFILE;

  int ch;
  while ((ch = getopt(argc, argv, "n:")) != -1)
  {
    switch (ch)
    {
      case 'n':
        sim_size = atoi(optarg);
        break;
      default:
        break;
    }
  }

  // zero is never a valid texture
  my_fluid_sim = create_fluid_sim(0, "fluid_kernel.cl", sim_size, 0.00001, 0.00001, flags);

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
