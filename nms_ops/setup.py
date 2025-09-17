from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "nms_ops",
        [
            "src/pybind.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_cmake_dir(),
        ],
        language='c++',
        cxx_std=14,
    ),
]

setup(
    name="nms_ops",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)