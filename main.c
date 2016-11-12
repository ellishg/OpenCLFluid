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

VEC_TYPE fluid_color = IS_A_DENSITY;

void toggle_source_color()
{
  fluid_color = (fluid_color == IS_A_DENSITY) ? IS_B_DENSITY : IS_A_DENSITY;
}

void on_clicked(float x, float y, int shift_held)
{
  const float dens_strength = 2.f;
  const float dens_radius = 1.f;
  const float vel_strength = 10.f;
  const float vel_radius = 0.05f;

  if (x_prev != -1 && y_prev != -1)
  {
    float u = x - x_prev;
    float v = y - y_prev;

    float dist_sqrd = u * u + v * v;

    if (dist_sqrd > 0)
    {
      float mag = vel_strength / sqrtf(dist_sqrd);

      if (shift_held)
      {
        enqueue_event(my_fluid_sim, x, y, dens_strength, dens_radius, fluid_color);
      }
      else {
        enqueue_event(my_fluid_sim, x, y, mag * u, vel_radius, IS_U_VELOCITY);
        enqueue_event(my_fluid_sim, x, y, mag * v, vel_radius, IS_V_VELOCITY);
      }
    }
  }
  x_prev = x;
  y_prev = y;
}

void on_release()
{
  x_prev = y_prev = -1;
}

void add_stream(FluidSim * fluid, float x, float y, float u, float v, float dens_strength, float vel_strength, VEC_TYPE source_type)
{
  const float radius = 0.05f;

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

    enqueue_event(fluid, x, y, dens_strength, radius, source_type);
    enqueue_event(fluid, x, y, u, radius, IS_U_VELOCITY);
    enqueue_event(fluid, x, y, v, radius, IS_V_VELOCITY);
  }
}

int main(int argc, char ** argv)
{

  FLAGS flags = 0;
  float viscosity = 0.0000001f;
  float diffusion_rate = 0.0001f;
  size_t sim_size = 256;

  int has_chosen_type = 0;

  int ch;
  while ((ch = getopt(argc, argv, "bpv:d:n:t:")) != -1)
  {
    switch (ch)
    {
      case 'b':
        flags |= F_DEBUG;
        break;
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
      default:
        break;
    }
  }

  if (!has_chosen_type)
  { //set defualt value
    flags |= F_USE_GPU;
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
    //add_stream(my_fluid_sim, 0.5f, 0.5f, -1.f, 1.f, 1.f, 1.f, IS_A_DENSITY);

    //add_stream(my_fluid_sim, 0.5f, 0.25f, 0.f, -1.f, 1.f, 2.f, IS_A_DENSITY);

    simulate_next_frame(my_fluid_sim, dt);

    render_window(my_window, 1.f / dt);

    do {
      poll_events(my_window, on_clicked, toggle_source_color, on_release);
    } while (SDL_GetTicks() - prev_time < 1000.f / MAX_FPS);
  }

  destroy_fluid_sim(my_fluid_sim);
  destroy_window(my_window);

  return 0;
}
