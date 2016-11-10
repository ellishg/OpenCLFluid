# OpenCLFluid
An efficient, yet powerful, interactive fluid simulation using OpenCL.

Make sure you have SDL2, OpenGL, and OpenCL installed to run.

# Usage
```Bash
./fluid -p (enables profiling) -t <CPU/GPU> -n <simulation size> -v <viscosity> -d <rate of diffusion>
```

The profile executable does not use SDL2 and will not render the simulation. It will only print usefull profiling information.

```Bash
./profile -t <CPU/GPU> -n <simulation size>
```
