#!/bin/bash
#SBATCH --job-name=benchmark_nbody
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module load cuda/12.4

echo "GPU Benchmark"
for n in 1000 10000 100000; do
    echo "Particles: $n"
    /usr/bin/time -f "%e seconds" ./nbody_cuda $n 1 10 10 128 > /dev/null
done

echo "CPU Benchmark"
for n in 1000 10000 100000; do
    echo "Particles: $n"
    /usr/bin/time -f "%e seconds" ./nbody $n 1 10 10 > /dev/null
done
