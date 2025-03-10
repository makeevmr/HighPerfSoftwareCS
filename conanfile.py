import os
from conan import ConanFile
from conan.tools.cmake import cmake_layout


class HighPerfSoftwareCSRecipe(ConanFile):
    settings = ("os", "compiler", "build_type", "arch")
    generators = ("CMakeDeps", "CMakeToolchain")

    def requirements(self):
        self.requires("openmpi/4.1.6")

    def layout(self):
        cmake_layout(self,
                     build_folder=(os.path.abspath(os.getcwd()) + "/conan"))