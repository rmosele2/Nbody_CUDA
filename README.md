# CUDA N-Body Simulation

This project implements a parallel N-body simulation in C++ using CUDA. It models gravitational interactions between particles and updates their motion over time. The project also includes a sequential CPU version for comparison and benchmarking.

## Files

- `nbody.cpp` – Sequential CPU implementation
- `nbody_cuda.cu` – Parallel CUDA implementation
- `Makefile` – Build targets for CPU and CUDA versions, benchmark targets
- `plot.py` – Python script to visualize particle states
- `solar.tsv` – Sample input (solar system model)

## Compilation

To build both versions:

```bash
make            # Builds CPU version (nbody)
make nbody_cuda # Builds CUDA version (nbody_cuda)