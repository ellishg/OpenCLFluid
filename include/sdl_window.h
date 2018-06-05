#ifndef __SDL_WINDOW
#define __SDL_WINDOW

#include <stdio.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

typedef struct window_t
{
  SDL_Window * sdl_window;
  SDL_GLContext gl_context;
  GLuint window_texture;
  GLint vertices[4];
  GLint tex_coords[8];
  GLubyte indices[6];
  size_t sim_size;
  int is_running;
  int shift_held;
} window_t;

window_t * create_window(size_t window_width, size_t window_height, size_t sim_size);

void destroy_window(window_t * window);

void render_window(window_t * window, int fps);

void poll_events(window_t * window, void (*on_clicked)(float x, float y, int shift_held), void (*toggle)(), void (*on_release)(void));

#endif
