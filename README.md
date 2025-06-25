# HighPerfSoftwareCS
Tasks for the course "Software for high-performance computer systems"

## How to build
[Install CUDA](https://www.cherryservers.com/blog/install-cuda-ubuntu)

Run `nvidia-smi --query-gpu=compute_cap --format=csv` to check compute capability

Change flag `-arch=sm_61` in file [CMakeLists.txt](src/CMakeLists.txt) according to the compute capability version

```
pip install conan
conan profile detect --name default
```
Make sure your profile has settings according to `conan_profile_settings.txt`

```
conan install . --build=missing -pr default
mkdir build && cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target {Task1|Task2|Task3|Task4}
```

## Run Task1

MPI execution:

```
mpiexec -n 4 ./src/Task1/MPI/mpi_hello_world
```

OpenMP execution:
```
export OMP_NUM_THREADS=4
./src/Task1/OpenMP/openmp_hello_world
```

CUDA execution:
```
./src/Task1/CUDA/cuda_hello_world
```

## Run Task2

MPI execution:

```
mpiexec -n <PROCESSES> ./src/Task2/MPI/mpi_array_sum <ARRAY_SIZE>
mpiexec -n 6 ./src/Task2/MPI/mpi_array_sum 10000
```

OpenMP execution:
```
./src/Task2/OpenMP/openmp_array_sum <THREADS> <ARRAY_SIZE>
./src/Task2/OpenMP/openmp_array_sum 6 10000
```

CUDA execution:
```
./src/Task2/CUDA/cuda_array_sum <ARRAY_SIZE>
./src/Task2/CUDA/cuda_array_sum 10000
```

## Run Task3

MPI execution:

```
mpiexec -n <PROCESSES> ./src/Task3/MPI/mpi_derivative
mpiexec -n 6 ./src/Task3/MPI/mpi_derivative
```

OpenMP execution:
```
./src/Task3/OpenMP/openmp_derivative <THREADS>
./src/Task3/OpenMP/openmp_derivative 6
```

CUDA execution:
```
./src/Task3/CUDA/cuda_derivative <BLOCKS> <X_SIZE> <Y_SIZE>
./src/Task3/CUDA/cuda_derivative 10 10 10
```

## Run Task4

OpenMP execution:
```
./src/Task4/OpenMP/openmp_matrix_composition <THREADS>
./src/Task4/OpenMP/openmp_matrix_composition 6
```

## References
- [Get started with mpi](https://www.paulnorvig.com/guides/using-mpi-with-c.html)
- [Get started with openmpi](https://curc.readthedocs.io/en/latest/programming/OpenMP-C.html)
- [Get started with cuda](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
