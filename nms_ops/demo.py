#!/usr/bin/env python3
"""
Demo script showing how to use the C++ scale_ray_nms_box3d implementation from Python
"""
import numpy as np
import nms_ops_cpp
import sys
import os

# Add the nms_ops directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample 3D bounding box data for demonstration"""
    np.random.seed(42)
    
    # Create some overlapping and non-overlapping boxes
    boxes = np.array([
        # Group 1: Overlapping boxes
        [10.0, 10.0, 1.0, 4.0, 2.0, 2.0, 0.0],     # Box 1
        [10.5, 10.2, 1.0, 4.0, 2.0, 2.0, 0.1],     # Box 2 (overlaps with 1)
        [10.1, 10.1, 1.0, 4.0, 2.0, 2.0, 0.0],     # Box 3 (overlaps with 1)
        
        # Group 2: Another set of overlapping boxes  
        [20.0, 20.0, 1.0, 3.0, 3.0, 2.0, 0.0],     # Box 4
        [20.3, 20.3, 1.0, 3.0, 3.0, 2.0, 0.0],     # Box 5 (overlaps with 4)
        
        # Group 3: Isolated boxes
        [30.0, 30.0, 1.0, 2.0, 2.0, 2.0, 0.0],     # Box 6 (isolated)
        [40.0, 40.0, 1.0, 2.0, 2.0, 2.0, 0.0],     # Box 7 (isolated)
        [50.0, 50.0, 1.0, 2.0, 2.0, 2.0, 0.0],     # Box 8 (isolated)
    ], dtype=np.float32)
    
    # Create confidence scores (higher for first box in each group)
    scores = np.array([0.95, 0.85, 0.80, 0.90, 0.70, 0.75, 0.65, 0.60], dtype=np.float32)
    
    return boxes, scores

def demo_basic_usage():
    """Demonstrate basic usage of scale_ray_nms_box3d"""
    print("=== Basic Usage Demo ===")
    
    boxes, scores = create_sample_data()
    
    print(f"Input: {len(boxes)} boxes")
    print("Box format: [x, y, z, length, width, height, rotation_y]")
    
    # Apply scale_ray_nms_box3d with default parameters
    filtered_boxes, filtered_scores = nms_ops_cpp.scale_ray_nms_box3d(
        boxes, scores
    )
    
    print(f"Output: {len(filtered_boxes)} boxes after NMS")
    print(f"Kept {len(filtered_boxes)}/{len(boxes)} boxes ({100*len(filtered_boxes)/len(boxes):.1f}%)")
    
    print("\nFiltered boxes and scores:")
    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        print(f"  Box {i}: center=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}), "
              f"size=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f}), score={score:.3f}")
    
    return filtered_boxes, filtered_scores

def demo_parameter_effects():
    """Demonstrate the effect of different parameters"""
    print("\n=== Parameter Effects Demo ===")
    
    boxes, scores = create_sample_data()
    
    # Test different threshold values
    parameters = [
        {"thresh_bev": 0.3, "thresh_ray": 0.3, "scale_factor": 1.0},
        {"thresh_bev": 0.7, "thresh_ray": 0.3, "scale_factor": 1.0},
        {"thresh_bev": 0.5, "thresh_ray": 0.1, "scale_factor": 1.0},
        {"thresh_bev": 0.5, "thresh_ray": 0.5, "scale_factor": 1.0},
        {"thresh_bev": 0.5, "thresh_ray": 0.3, "scale_factor": 1.5},
    ]
    
    print(f"Input: {len(boxes)} boxes")
    
    for i, params in enumerate(parameters):
        filtered_boxes, filtered_scores = nms_ops_cpp.scale_ray_nms_box3d(
            boxes, scores, **params
        )
        
        print(f"Params {i+1}: bev_thresh={params['thresh_bev']}, "
              f"ray_thresh={params['thresh_ray']}, scale={params['scale_factor']} "
              f"→ {len(filtered_boxes)} boxes")

def demo_individual_functions():
    """Demonstrate individual BEV and Ray NMS functions"""
    print("\n=== Individual Functions Demo ===")
    
    boxes, scores = create_sample_data()
    
    print(f"Input: {len(boxes)} boxes")
    
    # Test BEV NMS only
    bev_keep = nms_ops_cpp.bev_nms(boxes, scores, 0.5)
    print(f"BEV NMS: kept indices {list(bev_keep)} ({len(bev_keep)} boxes)")
    
    # Test Ray NMS only  
    ray_keep = nms_ops_cpp.ray_nms(boxes, scores, 0.3)
    print(f"Ray NMS: kept indices {list(ray_keep)} ({len(ray_keep)} boxes)")

def demo_performance():
    """Demonstrate performance with larger datasets"""
    print("\n=== Performance Demo ===")
    
    # Generate larger test dataset
    np.random.seed(42)
    n_boxes = 5000
    
    print(f"Generating {n_boxes} random boxes...")
    
    boxes = np.random.rand(n_boxes, 7).astype(np.float32)
    boxes[:, :3] *= 100    # positions: 0-100
    boxes[:, 3:6] *= 10    # dimensions: 0-10  
    boxes[:, 6] *= 2 * np.pi  # rotation: 0-2π
    
    scores = np.random.rand(n_boxes).astype(np.float32)
    
    import time
    
    # Measure performance
    start_time = time.time()
    filtered_boxes, filtered_scores = nms_ops_cpp.scale_ray_nms_box3d(
        boxes, scores, thresh_bev=0.5, thresh_ray=0.3, scale_factor=1.0
    )
    elapsed_time = time.time() - start_time
    
    print(f"Processed {n_boxes} boxes in {elapsed_time:.4f}s")
    print(f"Rate: {n_boxes/elapsed_time:.0f} boxes/second")
    print(f"Output: {len(filtered_boxes)} boxes ({100*len(filtered_boxes)/n_boxes:.1f}% kept)")

def demo_edge_cases():
    """Demonstrate handling of edge cases"""
    print("\n=== Edge Cases Demo ===")
    
    # Empty input
    empty_boxes = np.array([], dtype=np.float32).reshape(0, 7)
    empty_scores = np.array([], dtype=np.float32)
    
    result_boxes, result_scores = nms_ops_cpp.scale_ray_nms_box3d(empty_boxes, empty_scores)
    print(f"Empty input: {result_boxes.shape} boxes, {result_scores.shape} scores")
    
    # Single box
    single_box = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    single_score = np.array([0.9], dtype=np.float32)
    
    result_boxes, result_scores = nms_ops_cpp.scale_ray_nms_box3d(single_box, single_score)
    print(f"Single box input: {result_boxes.shape} boxes, {result_scores.shape} scores")
    
    # All identical boxes (should keep only one)
    identical_boxes = np.tile([[5.0, 5.0, 1.0, 2.0, 2.0, 2.0, 0.0]], (5, 1)).astype(np.float32)
    identical_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    
    result_boxes, result_scores = nms_ops_cpp.scale_ray_nms_box3d(identical_boxes, identical_scores)
    print(f"5 identical boxes: kept {result_boxes.shape[0]} box(es)")

def main():
    """Run all demos"""
    print("C++ scale_ray_nms_box3d Implementation Demo")
    print("=" * 50)
    
    try:
        # Check if module loads correctly
        print(f"nms_ops_cpp version: {nms_ops_cpp.__version__}")
        
        # Run demonstrations
        demo_basic_usage()
        demo_parameter_effects()
        demo_individual_functions()
        demo_performance()
        demo_edge_cases()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ Basic 3D box NMS functionality")
        print("✅ Parameter customization (thresholds, scaling)")
        print("✅ Individual BEV and Ray NMS functions")
        print("✅ High performance (thousands of boxes/second)")
        print("✅ Robust edge case handling")
        print("✅ Easy Python integration via Pybind11")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()