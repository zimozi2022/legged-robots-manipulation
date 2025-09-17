# NMS Operations

This module provides C++ implementations of Non-Maximum Suppression (NMS) algorithms with Python bindings using Pybind11.

## Features

- **BEV NMS**: Bird's Eye View Non-Maximum Suppression for 2D boxes
- **Ray NMS**: Ray-based Non-Maximum Suppression for 3D boxes  
- **Scale Ray NMS**: Scale-aware Ray-based NMS that considers object scales

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Build from source

#### Using setuptools:
```bash
cd nms_ops
python setup.py build_ext --inplace
```

#### Using CMake:
```bash
cd nms_ops
mkdir build
cd build
cmake ..
make
```

## Usage

### BEV NMS
```python
import numpy as np
import nms_ops

# 2D boxes format: [x1, y1, x2, y2]
boxes = np.array([
    [0.0, 0.0, 10.0, 10.0],
    [5.0, 5.0, 15.0, 15.0],
], dtype=np.float32)

scores = np.array([0.9, 0.8], dtype=np.float32)
keep_indices = nms_ops.bev_nms(boxes, scores, iou_threshold=0.5)
```

### Ray NMS
```python
import numpy as np
import nms_ops

# 3D boxes format: [x1, y1, z1, x2, y2, z2]
boxes = np.array([
    [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
    [5.0, 5.0, 5.0, 15.0, 15.0, 15.0],
], dtype=np.float32)

scores = np.array([0.9, 0.8], dtype=np.float32)
keep_indices = nms_ops.ray_nms(boxes, scores, iou_threshold=0.5)
```

### Scale Ray NMS
```python
import numpy as np
import nms_ops

# 3D boxes format: [x1, y1, z1, x2, y2, z2]
boxes = np.array([
    [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
    [5.0, 5.0, 5.0, 15.0, 15.0, 15.0],
], dtype=np.float32)

scores = np.array([0.9, 0.8], dtype=np.float32)
scales = np.array([1.0, 1.5], dtype=np.float32)

keep_indices = nms_ops.scale_ray_nms(
    boxes, scores, scales, 
    iou_threshold=0.5, 
    scale_threshold=0.1
)
```

## Testing

Run the test suite:
```bash
python test_nms.py
```

## Algorithm Details

### Scale Ray NMS

The Scale Ray NMS algorithm extends traditional NMS by incorporating object scale information:

1. **Scale Similarity Check**: Only applies NMS between boxes with similar scales
2. **Adaptive Thresholding**: Adjusts IoU threshold based on scale differences  
3. **3D Volume Calculation**: Uses 3D Intersection over Union for better spatial understanding

Key parameters:
- `iou_threshold`: Standard IoU threshold for suppression
- `scale_threshold`: Maximum relative scale difference for NMS consideration

This approach is particularly useful for multi-scale object detection where objects of different sizes should be preserved even if they spatially overlap.