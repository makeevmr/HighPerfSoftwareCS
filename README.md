# HighPerfSoftwareCS
Tasks for the course "Software for high-performance computer systems"

```
pip install conan
conan profile detect
conan profile detect --name GCC11
conan install . --build=missing -pr GCC11
mkdir build && cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
