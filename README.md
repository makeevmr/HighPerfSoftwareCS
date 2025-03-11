# HighPerfSoftwareCS
Tasks for the course "Software for high-performance computer systems"

## How to build
```
pip install conan
conan profile detect
conan profile detect --name default
conan install . --build=missing -pr default
mkdir build && cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . 
```

## Run Task1
MPI 

`mpiexec -n 4 ./mpi_hello_world`

OpenMP 
```
export OMP_NUM_THREADS=4
./openmp_hello_world
```

## References
[Get starded with mpi](https://www.paulnorvig.com/guides/using-mpi-with-c.html)
