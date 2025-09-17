from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os

# Get the absolute path to the nms_ops directory
nms_ops_dir = os.path.dirname(os.path.abspath(__file__))

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "nms_ops_cpp",
        [
            os.path.join(nms_ops_dir, "src", "nms_ops.cpp"),
            os.path.join(nms_ops_dir, "python_bindings", "pybind_module.cpp"),
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to our headers
            os.path.join(nms_ops_dir, "include"),
        ],
        cxx_std=17,
        language='c++',
    ),
]

setup(
    name="nms_ops",
    version="1.0.0",
    author="Auto-generated",
    author_email="",
    description="C++ implementation of NMS operations for 3D boxes",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pybind11>=2.6.0",
    ],
)