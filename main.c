/*
 * Honors project for CS241 by Ellis Hoag
 *
 * To compile on mac:
 * gcc main.c cl_fluid_sim.c sdl_window.c -framework OpenCL -framework OpenGL -framework SDL2 -ofluid
 */

#include <unistd.h>

#include "sdl_window.h"
#include "cl_fluid_sim.h"

#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600
#define MAX_FPS 60

extern char * optarg;

window_t * my_window;
FluidSim * my_fluid_sim;

float x_prev = -1;
float y_prev = -1;

void on_clicked(float x, float y, int shift_held)
{
  const float strength = 20.f;
  const float radius = 0.05f;

  if (shift_held)
  {
    enqueue_event(my_fluid_sim, x, y, 1, 1, IS_A_DENSITY);

  }
  else {
    if (x_prev != -1 && y_prev != -1)
    {
      float u = x - x_prev;
      float v = y - y_prev;

      float dist_sqrd = u * u + v * v;

      if (dist_sqrd > 0)
      {
        float mag = strength / sqrtf(dist_sqrd);

        enqueue_event(my_fluid_sim, x, y, mag * u, radius, IS_U_VELOCITY);
        enqueue_event(my_fluid_sim, x, y, mag * v, radius, IS_V_VELOCITY);
      }
    }
    x_prev = x;
    y_prev = y;
  }
}

void on_release()
{
  x_prev = y_prev = -1;
}

void add_stream(FluidSim * fluid, float x, float y, float u, float v, float dens_strength, float vel_strength, VEC_TYPE source_type)
{
  float u_offset = ((rand() % 100) - 50) / 30.f;
  float v_offset = ((rand() % 100) - 50) / 30.f;

  u += u_offset;
  v += v_offset;

  float dist_sqrd = u * u + v * v;

  if (dist_sqrd > 0)
  {
    float mag = vel_strength / sqrtf(dist_sqrd);

    u = mag * u;
    v = mag * v;

    enqueue_event(fluid, x, y, dens_strength, 2.f, source_type);
    enqueue_event(fluid, x, y, u, 2.f, IS_U_VELOCITY);
    enqueue_event(fluid, x, y, v, 2.f, IS_V_VELOCITY);
  }
}

int main(int argc, char ** argv)
{

  FLAGS flags = 0;
  float viscosity = 0.00001;
  float diffusion_rate = 0.000001;
  size_t sim_size = 256;

  int ch;
  while ((ch = getopt(argc, argv, "pv:d:n:")) != -1)
  {
    switch (ch)
    {
      case 'p':
        flags |= F_PROFILE;
        break;
      case 'v':
        viscosity = (float)atof(optarg);
        break;
      case 'd':
        diffusion_rate = (float)atof(optarg);
        break;
      case 'n':
        sim_size = atoi(optarg);
        break;
      default:
        break;
    }
  }

  my_window = create_window(WINDOW_WIDTH, WINDOW_HEIGHT, sim_size);

  if (!my_window)
  {
    fprintf(stderr, "Failed to create window!\n");
    return 2;
  }

  my_fluid_sim = create_fluid_sim(my_window->window_texture, "fluid_kernel.cl", sim_size, diffusion_rate, viscosity, flags);

  if (!my_fluid_sim)
  {
    fprintf(stderr, "Unable to create fluid_sim!\n");
    return 3;
  }

  Uint32 prev_time = SDL_GetTicks();

  while (my_window->is_running)
  {
    float dt = (SDL_GetTicks() - prev_time) / 1000.f;
    prev_time = SDL_GetTicks();

    add_stream(my_fluid_sim, 0.5f, 0.75f, 0.f, -1.f, 1.f, 1.f, IS_B_DENSITY);
    add_stream(my_fluid_sim, 0.25f, 0.25f, 1.f, 1.f, 1.f, 1.f, IS_A_DENSITY);
    add_stream(my_fluid_sim, 0.75f, 0.25f, -1.f, 1.f, 1.f, 1.f, IS_A_DENSITY);

    simulate_next_frame(my_fluid_sim, dt);

    render_window(my_window, 1.f / dt);

    do {
      poll_events(my_window, on_clicked, on_release);
    } while (SDL_GetTicks() - prev_time < 1000.f / MAX_FPS);
  }

  destroy_fluid_sim(my_fluid_sim);
  destroy_window(my_window);

  return 0;
}
