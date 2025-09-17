"""
Integration example for using NMS operations in legged robot manipulation tasks.
This demonstrates how the scale_ray_nms function can be used for object detection
in robotic manipulation scenarios.
"""

import numpy as np
import sys
import os

# Add the nms_ops module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'nms_ops'))

try:
    import nms_ops
except ImportError:
    print("Error: nms_ops module not found. Please build it first:")
    print("cd nms_ops && python setup.py build_ext --inplace")
    sys.exit(1)

def simulate_object_detection_scenario():
    """
    Simulate a scenario where a legged robot needs to detect and manipulate objects
    of different scales in its environment.
    """
    print("=== Legged Robot Object Detection Scenario ===")
    
    # Simulate detected objects in robot's workspace
    # Objects could be tools, obstacles, or manipulation targets
    
    # 3D bounding boxes in robot coordinate frame (meters)
    # Format: [x_min, y_min, z_min, x_max, y_max, z_max]
    detected_objects = np.array([
        # Large box/container
        [0.5, -0.2, 0.0, 0.8, 0.2, 0.3],
        
        # Medium tool overlapping with large box
        [0.6, -0.1, 0.05, 0.75, 0.1, 0.15],
        
        # Small screw/bolt near the tool
        [0.65, 0.0, 0.1, 0.68, 0.03, 0.13],
        
        # Another small component
        [0.7, -0.05, 0.08, 0.73, -0.02, 0.11],
        
        # Separate large object (another tool)
        [1.0, 0.3, 0.0, 1.3, 0.6, 0.2],
        
        # Medium object overlapping with second large object
        [1.1, 0.4, 0.05, 1.2, 0.5, 0.15],
    ], dtype=np.float32)
    
    # Detection confidence scores
    detection_scores = np.array([0.95, 0.88, 0.75, 0.82, 0.92, 0.78], dtype=np.float32)
    
    # Object scale information (could be derived from size or semantic class)
    # Large objects: 2.0+, Medium objects: 1.0-2.0, Small objects: <1.0
    object_scales = np.array([2.5, 1.5, 0.8, 0.9, 2.2, 1.3], dtype=np.float32)
    
    print(f"Detected {len(detected_objects)} objects in workspace")
    print(f"Detection scores: {detection_scores}")
    print(f"Object scales: {object_scales}")
    print()
    
    # Apply different NMS strategies
    
    # 1. Standard 3D NMS (ignores scale information)
    standard_keep = nms_ops.ray_nms(detected_objects, detection_scores, iou_threshold=0.3)
    
    # 2. Scale-aware NMS (preserves objects of different scales)
    scale_aware_keep = nms_ops.scale_ray_nms(
        detected_objects, detection_scores, object_scales,
        iou_threshold=0.3, scale_threshold=0.4
    )
    
    print("Standard Ray NMS Results:")
    print(f"  Kept {len(standard_keep)} objects: {standard_keep}")
    print(f"  Scales: {object_scales[standard_keep]}")
    print()
    
    print("Scale-Aware Ray NMS Results:")
    print(f"  Kept {len(scale_aware_keep)} objects: {scale_aware_keep}")
    print(f"  Scales: {object_scales[scale_aware_keep]}")
    print()
    
    # Show the practical difference
    only_in_scale_aware = set(scale_aware_keep) - set(standard_keep)
    if only_in_scale_aware:
        print(f"Objects preserved by scale-aware NMS: {list(only_in_scale_aware)}")
        for idx in only_in_scale_aware:
            print(f"  Object {idx}: scale={object_scales[idx]:.1f}, score={detection_scores[idx]:.2f}")
    else:
        print("Both methods produced the same results for this scenario.")
    
    return detected_objects, detection_scores, object_scales, scale_aware_keep

def manipulation_planning_example(objects, scores, scales, keep_indices):
    """
    Example of how the filtered objects could be used for manipulation planning
    """
    print("\n=== Manipulation Planning Example ===")
    
    final_objects = objects[keep_indices]
    final_scores = scores[keep_indices]
    final_scales = scales[keep_indices]
    
    # Sort by manipulation priority (combination of score and accessibility)
    # For this example, prioritize medium-scale objects with high confidence
    priority_scores = final_scores * (1.0 + 0.1 * (2.0 - np.abs(final_scales - 1.5)))
    manipulation_order = np.argsort(priority_scores)[::-1]
    
    print("Recommended manipulation order:")
    for i, obj_idx in enumerate(manipulation_order):
        orig_idx = keep_indices[obj_idx]
        print(f"  {i+1}. Object {orig_idx}: "
              f"scale={final_scales[obj_idx]:.1f}, "
              f"confidence={final_scores[obj_idx]:.2f}, "
              f"priority={priority_scores[obj_idx]:.2f}")
        
        # Show bounding box for robot path planning
        bbox = final_objects[obj_idx]
        center = (bbox[:3] + bbox[3:6]) / 2
        size = bbox[3:6] - bbox[:3]
        print(f"     Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"     Size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")

if __name__ == "__main__":
    print("NMS Operations - Legged Robot Integration Example")
    print("=" * 60)
    
    # Run the simulation
    objects, scores, scales, keep_indices = simulate_object_detection_scenario()
    
    # Show how results could be used for manipulation planning
    manipulation_planning_example(objects, scores, scales, keep_indices)
    
    print("\nExample completed successfully!")
    print("This demonstrates how scale_ray_nms can help robots distinguish")
    print("between objects of different scales even when they spatially overlap.")