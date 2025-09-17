#!/usr/bin/env python3
"""
Example usage of the C++ scale_ray_nms_box3d implementation
This script demonstrates how to integrate the NMS operations into your project
"""
import sys
import os

# Add the nms_ops directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nms_ops'))

import numpy as np
import nms_ops_cpp

def example_usage():
    """Example of how to use scale_ray_nms_box3d in a typical workflow"""
    
    # Simulate detection results from a 3D object detector
    print("Simulating 3D object detection results...")
    
    # Example: Vehicle detection results
    # Format: [x, y, z, length, width, height, rotation_y]
    detected_boxes = np.array([
        # Car detections
        [10.0, 5.0, 1.0, 4.5, 2.0, 1.8, 0.1],    # Car 1
        [10.2, 5.1, 1.0, 4.5, 2.0, 1.8, 0.1],    # Car 1 (duplicate)
        [15.0, 8.0, 1.0, 4.0, 1.8, 1.6, 0.0],    # Car 2
        
        # Pedestrian detections  
        [20.0, 3.0, 0.9, 0.6, 0.6, 1.8, 0.0],    # Person 1
        [20.1, 3.1, 0.9, 0.6, 0.6, 1.8, 0.0],    # Person 1 (duplicate)
        
        # Cyclist detection
        [25.0, 7.0, 1.0, 1.8, 0.8, 1.8, 0.5],    # Cyclist
        
    ], dtype=np.float32)
    
    # Confidence scores from the detector
    confidence_scores = np.array([0.95, 0.87, 0.82, 0.78, 0.71, 0.69], dtype=np.float32)
    
    print(f"Input: {len(detected_boxes)} detected objects")
    print("Box format: [x, y, z, length, width, height, rotation_y]")
    
    # Apply NMS to remove duplicate detections
    filtered_boxes, filtered_scores = nms_ops_cpp.scale_ray_nms_box3d(
        detected_boxes, 
        confidence_scores,
        thresh_bev=0.5,      # BEV IoU threshold  
        thresh_ray=0.3,      # 3D IoU threshold
        scale_factor=1.0     # No scaling
    )
    
    print(f"Output: {len(filtered_boxes)} objects after NMS")
    
    # Display results
    print("\nFiltered detections:")
    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        x, y, z, l, w, h, ry = box
        print(f"Object {i+1}: pos=({x:.1f}, {y:.1f}, {z:.1f}), "
              f"size=({l:.1f}×{w:.1f}×{h:.1f}), "
              f"rotation={ry:.2f}, confidence={score:.3f}")
    
    return filtered_boxes, filtered_scores

def benchmark_performance():
    """Benchmark the performance with larger datasets"""
    print("\nPerformance benchmark...")
    
    # Generate larger dataset for benchmarking
    np.random.seed(42)
    n_boxes = 2000
    
    boxes = np.random.rand(n_boxes, 7).astype(np.float32)
    boxes[:, :3] *= 50    # positions: 0-50 meters
    boxes[:, 3:6] *= 5    # dimensions: 0-5 meters
    boxes[:, 6] *= 2 * np.pi  # rotation: 0-2π
    
    scores = np.random.rand(n_boxes).astype(np.float32)
    
    import time
    start_time = time.time()
    
    filtered_boxes, filtered_scores = nms_ops_cpp.scale_ray_nms_box3d(
        boxes, scores, thresh_bev=0.7, thresh_ray=0.5
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"Processed {n_boxes} boxes in {elapsed_time:.4f}s")
    print(f"Processing rate: {n_boxes/elapsed_time:.0f} boxes/second")
    print(f"Kept {len(filtered_boxes)}/{n_boxes} boxes ({100*len(filtered_boxes)/n_boxes:.1f}%)")

def main():
    """Main demonstration"""
    print("C++ scale_ray_nms_box3d Integration Example")
    print("=" * 50)
    
    try:
        # Check if the module is available
        print(f"Using nms_ops_cpp version: {nms_ops_cpp.__version__}")
        
        # Run example usage
        example_usage()
        
        # Run performance benchmark
        benchmark_performance()
        
        print("\n✅ Integration example completed successfully!")
        print("\nTo integrate into your project:")
        print("1. Copy the nms_ops/ directory to your project")
        print("2. Run: cd nms_ops && python install.py")
        print("3. Import: import nms_ops_cpp")
        print("4. Use: nms_ops_cpp.scale_ray_nms_box3d(boxes, scores)")
        
    except ImportError as e:
        print(f"❌ Failed to import nms_ops_cpp: {e}")
        print("\nTo fix this issue:")
        print("1. cd nms_ops")
        print("2. python install.py")
        print("3. Run this script again")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()