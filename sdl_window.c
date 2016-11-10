#include "sdl_window.h"

const GLint vertices[] = {-1, 1,
                          1, 1,
                          1, -1,
                          -1, -1};

const GLint tex_coords[] = {0, 0,
                            1, 0,
                            1, 1,
                            0, 1};

window_t * create_window(size_t window_width, size_t window_height, size_t sim_size)
{

  if (SDL_Init(SDL_INIT_VIDEO))
  {
    fprintf(stderr, "Unable to initialize SDL: %s\n", SDL_GetError());
    return NULL;
  }

  SDL_Window * window = SDL_CreateWindow("Fluid2D", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_OPENGL);

  if (!window)
  {
    fprintf(stderr, "Unable to create window: %s\n", SDL_GetError());
    return NULL;
  }

  SDL_GLContext context = SDL_GL_CreateContext(window);

  if (context == 0)
  {
    fprintf(stderr, "Unable to create context: %s\n", SDL_GetError());
    return NULL;
  }

  GLuint texture;
  glEnable(GL_TEXTURE_2D);

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, sim_size, sim_size, 0, GL_RGBA, GL_FLOAT, NULL);

  GLenum err = glGetError();
  if (err != GL_NO_ERROR)
  {
    fprintf(stderr, "Window setup failed: %d\n", err);
    return NULL;
  }

  window_t * my_window = (window_t *)malloc(sizeof(window_t));
  my_window->sdl_window = window;
  my_window->gl_context = context;
  my_window->window_texture = texture;
  my_window->sim_size = sim_size;
  my_window->is_running = 1;
  my_window->shift_held = 0;

  return my_window;
}

void destroy_window(window_t * window)
{
  glDeleteTextures(1, &window->window_texture);
  SDL_GL_DeleteContext(window->gl_context);
  SDL_DestroyWindow(window->sdl_window);

  free(window);
}

void render_window(window_t * window, int fps)
{
  glBindTexture(GL_TEXTURE_2D, window->window_texture);

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(2, GL_INT, 0, vertices);

  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glTexCoordPointer(2, GL_INT, 0, tex_coords);

  glDrawArrays(GL_QUADS, 0, 4);

  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  const size_t max_title_length = 20;

  char title[max_title_length];

  snprintf(title, max_title_length, "Fluid2D FPS: %d", fps);

  SDL_SetWindowTitle(window->sdl_window, title);

  SDL_GL_SwapWindow(window->sdl_window);
}

void poll_events(window_t * window, void (*on_clicked)(float x, float y, int shift_held), void (*on_release)(void))
{
  int width, height;
  SDL_Event event;

  while (SDL_PollEvent(&event))
  {
    switch (event.type)
    {
      case SDL_QUIT:
        window->is_running = 0;
        break;
      case SDL_KEYDOWN:
        switch (event.key.keysym.sym)
        {
          case SDLK_ESCAPE:
            window->is_running = 0;
            break;
          case SDLK_LSHIFT:
            window->shift_held = 1;
            break;
          default:
            break;
        }
        break;
      case SDL_KEYUP:
        switch (event.key.keysym.sym)
        {
          case SDLK_LSHIFT:
            window->shift_held = 0;
            break;
          default:
            break;
        }
        break;
      case SDL_MOUSEMOTION:
      case SDL_MOUSEBUTTONDOWN:
        if (event.motion.state & SDL_BUTTON_LMASK)
        {
          SDL_GetWindowSize(window->sdl_window, &width, &height);
          on_clicked((float)event.button.x / width, (float)event.button.y / height, window->shift_held);
        }
        break;
      case SDL_MOUSEBUTTONUP:
        on_release();
        break;
      default:
        break;
    }
  }
}
