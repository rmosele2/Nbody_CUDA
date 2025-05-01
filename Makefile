# ======================
# Compiler Configuration
# ======================

CXX = g++
CXXFLAGS = -O3

NVCC = nvcc
NVCCFLAGS = -arch=sm_61 -O2

# ======================
# Executables
# ======================

nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) nbody.cpp -o nbody

nbody_cuda: nbody_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o nbody_cuda nbody_cuda.cu

# ======================
# Output Targets
# ======================

solar.out: nbody
	date
	./nbody planet 200 5000000 10000 > solar.out
	date

solar.pdf: solar.out
	python3 plot.py solar.out solar.pdf 1000 

random.out: nbody
	date
	./nbody 1000 1 10000 100 > random.out
	date

cuda.out: nbody_cuda
	date
	./nbody_cuda 1000 1 1000 100 128 > cuda.out
	date

# ======================
# Benchmarking Targets
# ======================

benchmark_cpu: nbody
	@echo "Benchmarking CPU version"
	@for n in 1000 10000 100000; do \
		echo "Particles: $$n"; \
		TIMEFORMAT="%3R seconds"; \
		time ./nbody $$n 1 10 10 > /dev/null; \
	done

benchmark_gpu: nbody_cuda
	@echo "Benchmarking GPU version"
	@for n in 1000 10000 100000; do \
		echo "Particles: $$n"; \
		TIMEFORMAT="%3R seconds"; \
		time ./nbody_cuda $$n 1 10 10 128 > /dev/null; \
	done

# ======================
# Cleanup
# ======================

clean:
	rm -f nbody nbody_cuda *.out *.pdf
