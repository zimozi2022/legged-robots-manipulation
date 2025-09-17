"""
Comprehensive examples for NMS operations usage
"""
import numpy as np
import nms_ops

def example_bev_nms():
    """
    Example usage of Bird's Eye View NMS for 2D object detection
    """
    print("=== BEV NMS Example ===")
    
    # Sample 2D bounding boxes from a typical object detection scenario
    # Format: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
    boxes = np.array([
        [100, 100, 200, 200],  # Object 1: high score
        [150, 150, 250, 250],  # Object 2: overlaps with Object 1
        [300, 300, 400, 400],  # Object 3: separate object, highest score
        [110, 110, 210, 210],  # Object 4: heavily overlaps with Object 1
        [500, 100, 600, 200],  # Object 5: separate object
    ], dtype=np.float32)
    
    # Confidence scores for each detection
    scores = np.array([0.85, 0.75, 0.95, 0.65, 0.80], dtype=np.float32)
    
    print(f"Input: {len(boxes)} boxes")
    print(f"Scores: {scores}")
    
    # Apply BEV NMS with IoU threshold of 0.5
    keep_indices = nms_ops.bev_nms(boxes, scores, iou_threshold=0.5)
    
    print(f"After NMS: {len(keep_indices)} boxes kept")
    print(f"Keep indices: {keep_indices}")
    print(f"Final boxes:\n{boxes[keep_indices]}")
    print(f"Final scores: {scores[keep_indices]}")
    print()

def example_ray_nms():
    """
    Example usage of Ray NMS for 3D object detection
    """
    print("=== Ray NMS Example ===")
    
    # Sample 3D bounding boxes for point cloud object detection
    # Format: [x1, y1, z1, x2, y2, z2] where min and max coordinates
    boxes = np.array([
        [0, 0, 0, 2, 2, 2],      # Car 1
        [1, 1, 0, 3, 3, 2],      # Car 2: overlaps with Car 1
        [10, 10, 0, 12, 12, 2],  # Car 3: separate
        [0.5, 0.5, 0, 2.5, 2.5, 2], # Car 4: heavily overlaps with Car 1
        [5, 5, 0, 7, 7, 2],      # Car 5: separate
    ], dtype=np.float32)
    
    scores = np.array([0.90, 0.85, 0.95, 0.70, 0.88], dtype=np.float32)
    
    print(f"Input: {len(boxes)} 3D boxes")
    print(f"Scores: {scores}")
    
    # Apply Ray NMS with IoU threshold of 0.3
    keep_indices = nms_ops.ray_nms(boxes, scores, iou_threshold=0.3)
    
    print(f"After NMS: {len(keep_indices)} boxes kept")
    print(f"Keep indices: {keep_indices}")
    print(f"Final boxes:\n{boxes[keep_indices]}")
    print(f"Final scores: {scores[keep_indices]}")
    print()

def example_scale_ray_nms():
    """
    Example usage of Scale-aware Ray NMS for multi-scale 3D object detection
    """
    print("=== Scale Ray NMS Example ===")
    
    # 3D boxes representing objects of different scales
    boxes = np.array([
        [0, 0, 0, 4, 4, 2],      # Large car
        [1, 1, 0, 3, 3, 1.5],    # Medium car, overlaps spatially
        [10, 10, 0, 12, 14, 2],  # Large truck
        [0.5, 0.5, 0, 2, 2, 1],  # Small car, overlaps with large car
        [11, 11, 0, 12.5, 12.5, 1], # Small object near truck
        [20, 20, 0, 24, 24, 2],  # Another large car, separate
    ], dtype=np.float32)
    
    scores = np.array([0.90, 0.85, 0.95, 0.80, 0.75, 0.92], dtype=np.float32)
    
    # Scale values representing object sizes
    scales = np.array([2.0, 1.5, 3.0, 1.0, 0.8, 2.0], dtype=np.float32)
    
    print(f"Input: {len(boxes)} 3D boxes")
    print(f"Scores: {scores}")
    print(f"Scales: {scales}")
    
    # Apply Scale Ray NMS
    # iou_threshold: standard IoU threshold
    # scale_threshold: maximum relative scale difference for NMS consideration
    keep_indices = nms_ops.scale_ray_nms(
        boxes, scores, scales, 
        iou_threshold=0.3, 
        scale_threshold=0.3
    )
    
    print(f"After Scale Ray NMS: {len(keep_indices)} boxes kept")
    print(f"Keep indices: {keep_indices}")
    print(f"Final boxes:\n{boxes[keep_indices]}")
    print(f"Final scores: {scores[keep_indices]}")
    print(f"Final scales: {scales[keep_indices]}")
    print()

def example_comparison():
    """
    Compare different NMS methods on the same data
    """
    print("=== NMS Methods Comparison ===")
    
    # Convert 3D boxes to 2D for BEV comparison (using x,y coordinates only)
    boxes_3d = np.array([
        [0, 0, 0, 2, 2, 2],
        [1, 1, 0, 3, 3, 2],
        [10, 10, 0, 12, 12, 2],
        [0.5, 0.5, 0, 2.5, 2.5, 2],
    ], dtype=np.float32)
    
    boxes_2d = boxes_3d[:, [0, 1, 3, 4]]  # Extract x1, y1, x2, y2
    scores = np.array([0.90, 0.85, 0.95, 0.70], dtype=np.float32)
    scales = np.array([1.5, 1.2, 2.0, 1.0], dtype=np.float32)
    
    print(f"Input data: {len(boxes_3d)} boxes")
    print(f"Scores: {scores}")
    print(f"Scales: {scales}")
    
    # Compare results
    bev_keep = nms_ops.bev_nms(boxes_2d, scores, 0.5)
    ray_keep = nms_ops.ray_nms(boxes_3d, scores, 0.5)
    scale_keep = nms_ops.scale_ray_nms(boxes_3d, scores, scales, 0.5, 0.2)
    
    print(f"BEV NMS keeps: {bev_keep} ({len(bev_keep)} boxes)")
    print(f"Ray NMS keeps: {ray_keep} ({len(ray_keep)} boxes)")
    print(f"Scale Ray NMS keeps: {scale_keep} ({len(scale_keep)} boxes)")
    print()

if __name__ == "__main__":
    print("NMS Operations - Comprehensive Examples")
    print("=" * 50)
    
    try:
        example_bev_nms()
        example_ray_nms()
        example_scale_ray_nms()
        example_comparison()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure to build the extension first: python setup.py build_ext --inplace")