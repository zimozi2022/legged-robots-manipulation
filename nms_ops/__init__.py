"""
NMS Operations Package

This package provides C++ implementations of Non-Maximum Suppression algorithms
with Python bindings using Pybind11.

Available functions:
- bev_nms: Bird's Eye View Non-Maximum Suppression for 2D boxes
- ray_nms: Ray-based Non-Maximum Suppression for 3D boxes
- scale_ray_nms: Scale-aware Ray-based NMS for multi-scale detection

Author: Generated for legged-robots-manipulation project
"""

try:
    from .nms_ops import bev_nms, ray_nms, scale_ray_nms
    __all__ = ['bev_nms', 'ray_nms', 'scale_ray_nms']
except ImportError:
    # Module not built yet
    print("NMS operations module not built. Run: python setup.py build_ext --inplace")
    __all__ = []

__version__ = "1.0.0"