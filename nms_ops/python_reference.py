"""
Python reference implementation of scale_ray_nms_box3d
This implementation serves as the reference for the C++ version
"""
import numpy as np
from typing import Tuple, List


def bev_nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    """
    Bird's Eye View Non-Maximum Suppression
    
    Args:
        boxes: (N, 7) array of 3D boxes [x, y, z, l, w, h, ry]
        scores: (N,) array of confidence scores
        thresh: IoU threshold for suppression
        
    Returns:
        List of indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to BEV (Bird's Eye View) representation
    x1 = boxes[:, 0] - boxes[:, 3] / 2  # left
    y1 = boxes[:, 1] - boxes[:, 4] / 2  # top  
    x2 = boxes[:, 0] + boxes[:, 3] / 2  # right
    y2 = boxes[:, 1] + boxes[:, 4] / 2  # bottom
    
    areas = boxes[:, 3] * boxes[:, 4]  # length * width
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    
    return keep


def ray_nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    """
    Ray-based Non-Maximum Suppression for 3D boxes
    
    Args:
        boxes: (N, 7) array of 3D boxes [x, y, z, l, w, h, ry]
        scores: (N,) array of confidence scores  
        thresh: IoU threshold for suppression
        
    Returns:
        List of indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # For ray-based NMS, we consider the 3D overlap including height
    x1 = boxes[:, 0] - boxes[:, 3] / 2
    y1 = boxes[:, 1] - boxes[:, 4] / 2
    z1 = boxes[:, 2] - boxes[:, 5] / 2
    x2 = boxes[:, 0] + boxes[:, 3] / 2
    y2 = boxes[:, 1] + boxes[:, 4] / 2
    z2 = boxes[:, 2] + boxes[:, 5] / 2
    
    volumes = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]  # l * w * h
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Calculate 3D IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        zz2 = np.minimum(z2[i], z2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        d = np.maximum(0.0, zz2 - zz1)
        inter = w * h * d
        
        iou = inter / (volumes[i] + volumes[order[1:]] - inter)
        
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    
    return keep


def scale_ray_nms_box3d(boxes: np.ndarray, 
                       scores: np.ndarray,
                       thresh_bev: float = 0.5,
                       thresh_ray: float = 0.3,
                       scale_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale and apply ray-based NMS on 3D boxes
    
    This function first scales the boxes, then applies BEV NMS followed by Ray NMS
    to filter overlapping 3D bounding boxes.
    
    Args:
        boxes: (N, 7) numpy array of 3D boxes [x, y, z, l, w, h, ry]
               where (x,y,z) is center, (l,w,h) are dimensions, ry is rotation
        scores: (N,) numpy array of confidence scores
        thresh_bev: IoU threshold for BEV NMS
        thresh_ray: IoU threshold for Ray NMS  
        scale_factor: Factor to scale box dimensions
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores) after NMS
    """
    if len(boxes) == 0:
        return np.array([]).reshape(0, 7), np.array([])
    
    # Scale the boxes
    scaled_boxes = boxes.copy()
    scaled_boxes[:, 3:6] *= scale_factor  # Scale length, width, height
    
    # Apply BEV NMS first
    bev_keep = bev_nms(scaled_boxes, scores, thresh_bev)
    
    if len(bev_keep) == 0:
        return np.array([]).reshape(0, 7), np.array([])
    
    # Apply Ray NMS on BEV results
    bev_boxes = scaled_boxes[bev_keep]
    bev_scores = scores[bev_keep]
    
    ray_keep = ray_nms(bev_boxes, bev_scores, thresh_ray)
    
    # Map back to original indices
    final_keep = [bev_keep[i] for i in ray_keep]
    
    return boxes[final_keep], scores[final_keep]


# Test function
def test_scale_ray_nms_box3d():
    """Test the scale_ray_nms_box3d function"""
    # Create test data
    boxes = np.array([
        [1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 1
        [1.1, 1.1, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 2 (overlap with 1)
        [5.0, 5.0, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 3 (distant)
    ])
    scores = np.array([0.9, 0.8, 0.7])
    
    filtered_boxes, filtered_scores = scale_ray_nms_box3d(
        boxes, scores, thresh_bev=0.5, thresh_ray=0.3, scale_factor=1.2
    )
    
    print("Original boxes shape:", boxes.shape)
    print("Filtered boxes shape:", filtered_boxes.shape)
    print("Original scores:", scores)
    print("Filtered scores:", filtered_scores)
    
    return filtered_boxes, filtered_scores


if __name__ == "__main__":
    test_scale_ray_nms_box3d()