# HighPerfSoftwareCS
Tasks for the course "Software for high-performance computer systems"

## How to build
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

`mpiexec -n 4 ./src/Task1/mpi_hello_world`

OpenMP execution:
```
export OMP_NUM_THREADS=4
./src/Task1/openmp_hello_world
```

## References
[Get starded with mpi](https://www.paulnorvig.com/guides/using-mpi-with-c.html)
