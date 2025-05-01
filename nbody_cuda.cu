#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define G 6.67e-11 // Gravitational constant
#define SOFTENING 1e-9

__global__ void compute_forces(double *x, double *y, double *z, double *mass, double *fx, double *fy, double *fz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    double xi = x[i];
    double yi = y[i];
    double zi = z[i];

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    for (int j = 0; j < n; j++) {
        if (i == j) {
            continue;
        }
        double dx = x[j] - xi;
        double dy = y[j] - yi;
        double dz = z[j] - zi;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        double F = G * mass[i] * mass[j] * invDist3;
        fxi += F * dx;
        fyi += F * dy;
        fzi += F * dz;
    }
    fx[i] = fxi;
    fy[i] = fyi;
    fz[i] = fzi;
}

__global__ void update_positions(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *fx, double *fy, double *fz, double *mass, double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;

    vx[i] += fx[i] / mass[i] * dt;
    vy[i] += fy[i] / mass[i] * dt;
    vz[i] += fz[i] / mass[i] * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

void dump_state(double *mass, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *fx, double *fy, double *fz, int n) {
    std::cout << n << '\t';
    for (int i = 0; i < n; i++) {
        std::cout << mass[i] << '\t' << y[i] << '\t' << z[i] << '\t' << vx[i] << '\t' << vy[i] << '\t' << vz[i] << '\t' << fx[i] << '\t' << fy[i] << '\t' << fz[i] << 't';
    }
    std::cout << '\n';
}

int main (int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <n> <dt> <steps> <print_every> <block_size>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    double dt = atof(argv[2]);
    int steps = atoi(argv[3]);
    int print_every = atoi(argv[4]);
    int block_size = atoi(argv[5]);

    size_t bytes = n * sizeof(double);

    double *h_x = new double[n], *h_y = new double[n], *h_z = new double[n];
    double *h_vx = new double[n], *h_vy = new double[n], *h_vz = new double[n];
    double *h_fx = new double[n], *h_fy = new double[n], *h_fz = new double[n];
    double *h_mass = new double[n];

    for (int i = 0; i < n; i++) {
        h_mass[i] = 1.0;
        h_x[i] = drand48(); h_y[i] = drand48(); h_z[i] = 0.0;
        h_vx[i] = 0.0; h_vy[i] = 0.0; h_vz[i] = 0.0;
    }

    double *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz, *d_mass;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes); cudaMalloc(&d_z, bytes);
    cudaMalloc(&d_vx, bytes); cudaMalloc(&d_vy, bytes); cudaMalloc(&d_vz, bytes);
    cudaMalloc(&d_fx, bytes); cudaMalloc(&d_fy, bytes); cudaMalloc(&d_fz, bytes);
    cudaMalloc(&d_mass, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, bytes, cudaMemcpyHostToDevice);

    int grid_size = (n + block_size - 1) / block_size;

    for (int step = 0; step < steps; step++) {
        compute_forces<<<grid_size, block_size>>>(d_x, d_y, d_z, d_mass, d_fx, d_fy, d_fz, n);
        update_positions<<<grid_size, block_size>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, n);

        if (step % print_every == 0) {
            cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vx, d_vx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy, d_vy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vz, d_vz, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fx, d_fx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy, d_fy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fz, d_fz, bytes, cudaMemcpyDeviceToHost);

            dump_state(h_mass,h_x,h_y,h_z,h_vx,h_vy,h_vz,h_fx,h_fy,h_fz,n);
        }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    cudaFree(d_mass);

    delete[] h_x; delete[] h_y; delete[] h_z;
    delete[] h_vx; delete[] h_vy; delete[] h_vz;
    delete[] h_fx; delete[] h_fy; delete[] h_fz;
    delete[] h_mass;

    return 0;

}
