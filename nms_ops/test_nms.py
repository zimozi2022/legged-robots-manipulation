"""
Python examples and tests for NMS operations
"""
import numpy as np

def test_bev_nms():
    """Test Bird's Eye View NMS"""
    try:
        import nms_ops
        
        # Create sample 2D boxes [x1, y1, x2, y2]
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],   # overlapping with first box
            [20.0, 20.0, 30.0, 30.0], # non-overlapping
            [1.0, 1.0, 11.0, 11.0],   # highly overlapping with first box
        ], dtype=np.float32)
        
        scores = np.array([0.9, 0.8, 0.95, 0.7], dtype=np.float32)
        iou_threshold = 0.5
        
        keep_indices = nms_ops.bev_nms(boxes, scores, iou_threshold)
        print(f"BEV NMS - Keep indices: {keep_indices}")
        print(f"Kept boxes: {boxes[keep_indices]}")
        print(f"Kept scores: {scores[keep_indices]}")
        
        return True
    except ImportError:
        print("nms_ops module not built yet. Run: cd nms_ops && python setup.py build_ext --inplace")
        return False
    except Exception as e:
        print(f"BEV NMS test failed: {e}")
        return False

def test_ray_nms():
    """Test Ray-based NMS"""
    try:
        import nms_ops
        
        # Create sample 3D boxes [x1, y1, z1, x2, y2, z2]
        boxes = np.array([
            [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0, 15.0, 15.0, 15.0],   # overlapping with first box
            [20.0, 20.0, 20.0, 30.0, 30.0, 30.0], # non-overlapping
            [1.0, 1.0, 1.0, 11.0, 11.0, 11.0],   # highly overlapping with first box
        ], dtype=np.float32)
        
        scores = np.array([0.9, 0.8, 0.95, 0.7], dtype=np.float32)
        iou_threshold = 0.5
        
        keep_indices = nms_ops.ray_nms(boxes, scores, iou_threshold)
        print(f"Ray NMS - Keep indices: {keep_indices}")
        print(f"Kept boxes: {boxes[keep_indices]}")
        print(f"Kept scores: {scores[keep_indices]}")
        
        return True
    except ImportError:
        print("nms_ops module not built yet. Run: cd nms_ops && python setup.py build_ext --inplace")
        return False
    except Exception as e:
        print(f"Ray NMS test failed: {e}")
        return False

def test_scale_ray_nms():
    """Test Scale-aware Ray-based NMS"""
    try:
        import nms_ops
        
        # Create sample 3D boxes [x1, y1, z1, x2, y2, z2]
        boxes = np.array([
            [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0, 15.0, 15.0, 15.0],   # overlapping with first box
            [20.0, 20.0, 20.0, 30.0, 30.0, 30.0], # non-overlapping
            [1.0, 1.0, 1.0, 11.0, 11.0, 11.0],   # highly overlapping with first box
            [2.0, 2.0, 2.0, 8.0, 8.0, 8.0],     # different scale, overlapping
        ], dtype=np.float32)
        
        scores = np.array([0.9, 0.8, 0.95, 0.7, 0.85], dtype=np.float32)
        scales = np.array([1.0, 1.1, 2.0, 1.0, 0.5], dtype=np.float32)  # different scales
        iou_threshold = 0.5
        scale_threshold = 0.2
        
        keep_indices = nms_ops.scale_ray_nms(boxes, scores, scales, iou_threshold, scale_threshold)
        print(f"Scale Ray NMS - Keep indices: {keep_indices}")
        print(f"Kept boxes: {boxes[keep_indices]}")
        print(f"Kept scores: {scores[keep_indices]}")
        print(f"Kept scales: {scales[keep_indices]}")
        
        return True
    except ImportError:
        print("nms_ops module not built yet. Run: cd nms_ops && python setup.py build_ext --inplace")
        return False
    except Exception as e:
        print(f"Scale Ray NMS test failed: {e}")
        return False

def run_performance_test():
    """Run performance test with larger datasets"""
    try:
        import nms_ops
        import time
        
        # Generate large dataset
        np.random.seed(42)
        num_boxes = 1000
        
        # 3D boxes
        centers = np.random.uniform(-50, 50, (num_boxes, 3))
        sizes = np.random.uniform(1, 10, (num_boxes, 3))
        boxes = np.concatenate([
            centers - sizes/2,  # min coordinates
            centers + sizes/2   # max coordinates
        ], axis=1).astype(np.float32)
        
        scores = np.random.uniform(0.1, 1.0, num_boxes).astype(np.float32)
        scales = np.random.uniform(0.5, 2.0, num_boxes).astype(np.float32)
        
        # Test performance
        start_time = time.time()
        keep_indices = nms_ops.scale_ray_nms(boxes, scores, scales, 0.5, 0.2)
        end_time = time.time()
        
        print(f"Performance test:")
        print(f"Input boxes: {num_boxes}")
        print(f"Output boxes: {len(keep_indices)}")
        print(f"Processing time: {(end_time - start_time)*1000:.2f} ms")
        
        return True
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing NMS Operations")
    print("=" * 50)
    
    print("\n1. Testing BEV NMS:")
    test_bev_nms()
    
    print("\n2. Testing Ray NMS:")
    test_ray_nms()
    
    print("\n3. Testing Scale Ray NMS:")
    test_scale_ray_nms()
    
    print("\n4. Performance Test:")
    run_performance_test()