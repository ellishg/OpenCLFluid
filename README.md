# OpenCLFluid
An efficient, yet powerful, interactive fluid simulation using OpenCL.
http://www.intpowertechcorp.com/GDC03.pdf

Make sure you have SDL2, OpenGL, and OpenCL installed to run.

# Usage
```Bash
./fluid [-pb] [-t <CPU/GPU>] [-n <simulation size>] [-v <viscosity>] [-d <rate of diffusion>]
```

-p enables profiling.

-b enables some debugging information.

-t chooses if you want to try to run on the CPU or GPU (defaults to GPU).

-n sets the simulations size (defaults to 256). This will generate a n x n simulation grid. Note that the simulation size must be a power of 2.

-v sets the viscosity of the fluid (defaults to 0.0000001f).

-d sets the rate of diffusion of the fluid (defaults to 0.0001f).


The profile executable does not use SDL2 and will not render the simulation. It will only print useful profiling information.

```Bash
./profile -t <CPU/GPU> -n <simulation size>
```
