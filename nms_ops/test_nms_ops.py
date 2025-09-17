#!/usr/bin/env python3
"""
Test script to compare C++ implementation with Python reference
"""
import numpy as np
import sys
import os

# Add the nms_ops directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import both implementations
from python_reference import scale_ray_nms_box3d as py_scale_ray_nms_box3d
from python_reference import bev_nms as py_bev_nms, ray_nms as py_ray_nms
import nms_ops_cpp

def test_basic_functionality():
    """Test basic functionality with simple test cases"""
    print("=== Testing Basic Functionality ===")
    
    # Create test data
    boxes = np.array([
        [1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 1
        [1.1, 1.1, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 2 (overlap with 1)
        [5.0, 5.0, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 3 (distant)
        [5.1, 5.1, 0.5, 2.0, 1.0, 1.0, 0.0],  # box 4 (overlap with 3)
    ], dtype=np.float32)
    
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    
    print(f"Input boxes shape: {boxes.shape}")
    print(f"Input scores: {scores}")
    
    # Test Python implementation
    py_result_boxes, py_result_scores = py_scale_ray_nms_box3d(
        boxes, scores, thresh_bev=0.5, thresh_ray=0.3, scale_factor=1.2
    )
    
    # Test C++ implementation
    cpp_result_boxes, cpp_result_scores = nms_ops_cpp.scale_ray_nms_box3d(
        boxes, scores, thresh_bev=0.5, thresh_ray=0.3, scale_factor=1.2
    )
    
    print(f"\nPython result boxes shape: {py_result_boxes.shape}")
    print(f"Python result scores: {py_result_scores}")
    print(f"C++ result boxes shape: {cpp_result_boxes.shape}")
    print(f"C++ result scores: {cpp_result_scores}")
    
    # Compare results
    if py_result_boxes.shape == cpp_result_boxes.shape:
        boxes_close = np.allclose(py_result_boxes, cpp_result_boxes, rtol=1e-5)
        scores_close = np.allclose(py_result_scores, cpp_result_scores, rtol=1e-5)
        
        print(f"\nBoxes match: {boxes_close}")
        print(f"Scores match: {scores_close}")
        
        if boxes_close and scores_close:
            print("✅ Basic functionality test PASSED")
            return True
        else:
            print("❌ Basic functionality test FAILED")
            print("Python boxes:", py_result_boxes)
            print("C++ boxes:", cpp_result_boxes)
            return False
    else:
        print("❌ Basic functionality test FAILED - different output shapes")
        return False

def test_individual_nms_functions():
    """Test individual BEV and Ray NMS functions"""
    print("\n=== Testing Individual NMS Functions ===")
    
    boxes = np.array([
        [0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 2.0, 2.0, 1.0, 0.0],  # overlaps with first
        [3.0, 3.0, 0.0, 2.0, 2.0, 1.0, 0.0],  # separate
    ], dtype=np.float32)
    
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    
    # Test BEV NMS
    py_bev_keep = py_bev_nms(boxes, scores, 0.5)
    cpp_bev_keep = nms_ops_cpp.bev_nms(boxes, scores, 0.5)
    
    print(f"Python BEV NMS keep: {py_bev_keep}")
    print(f"C++ BEV NMS keep: {list(cpp_bev_keep)}")
    
    bev_match = set(py_bev_keep) == set(cpp_bev_keep)
    print(f"BEV NMS results match: {bev_match}")
    
    # Test Ray NMS
    py_ray_keep = py_ray_nms(boxes, scores, 0.3)
    cpp_ray_keep = nms_ops_cpp.ray_nms(boxes, scores, 0.3)
    
    print(f"Python Ray NMS keep: {py_ray_keep}")
    print(f"C++ Ray NMS keep: {list(cpp_ray_keep)}")
    
    ray_match = set(py_ray_keep) == set(cpp_ray_keep)
    print(f"Ray NMS results match: {ray_match}")
    
    if bev_match and ray_match:
        print("✅ Individual NMS functions test PASSED")
        return True
    else:
        print("❌ Individual NMS functions test FAILED")
        return False

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    # Empty input
    empty_boxes = np.array([], dtype=np.float32).reshape(0, 7)
    empty_scores = np.array([], dtype=np.float32)
    
    py_empty_boxes, py_empty_scores = py_scale_ray_nms_box3d(empty_boxes, empty_scores)
    cpp_empty_boxes, cpp_empty_scores = nms_ops_cpp.scale_ray_nms_box3d(empty_boxes, empty_scores)
    
    print(f"Python empty boxes shape: {py_empty_boxes.shape}")
    print(f"Python empty scores shape: {py_empty_scores.shape}")
    print(f"C++ empty boxes shape: {cpp_empty_boxes.shape}")
    print(f"C++ empty scores shape: {cpp_empty_scores.shape}")
    
    empty_test = (py_empty_boxes.shape == cpp_empty_boxes.shape == (0, 7) and 
                  py_empty_scores.shape == cpp_empty_scores.shape == (0,))
    
    print(f"Empty input test: {empty_test}")
    
    # Single box
    single_box = np.array([[1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    single_score = np.array([0.9], dtype=np.float32)
    
    py_single_boxes, py_single_scores = py_scale_ray_nms_box3d(single_box, single_score)
    cpp_single_boxes, cpp_single_scores = nms_ops_cpp.scale_ray_nms_box3d(single_box, single_score)
    
    single_test = (np.allclose(py_single_boxes, cpp_single_boxes) and 
                   np.allclose(py_single_scores, cpp_single_scores))
    
    print(f"Single box test: {single_test}")
    
    if empty_test and single_test:
        print("✅ Edge cases test PASSED")
        return True
    else:
        print("❌ Edge cases test FAILED")
        return False

def performance_test():
    """Performance comparison between Python and C++"""
    print("\n=== Performance Test ===")
    
    # Generate larger test data
    np.random.seed(42)
    n_boxes = 1000
    
    boxes = np.random.rand(n_boxes, 7).astype(np.float32)
    boxes[:, :3] *= 100  # positions
    boxes[:, 3:6] *= 5   # dimensions
    boxes[:, 6] *= 2 * np.pi  # rotation
    
    scores = np.random.rand(n_boxes).astype(np.float32)
    
    print(f"Testing with {n_boxes} boxes")
    
    import time
    
    # Python version
    start_time = time.time()
    py_result_boxes, py_result_scores = py_scale_ray_nms_box3d(boxes, scores)
    py_time = time.time() - start_time
    
    # C++ version
    start_time = time.time()
    cpp_result_boxes, cpp_result_scores = nms_ops_cpp.scale_ray_nms_box3d(boxes, scores)
    cpp_time = time.time() - start_time
    
    print(f"Python time: {py_time:.4f}s")
    print(f"C++ time: {cpp_time:.4f}s")
    print(f"Speedup: {py_time/cpp_time:.2f}x")
    
    print(f"Python result: {py_result_boxes.shape[0]} boxes")
    print(f"C++ result: {cpp_result_boxes.shape[0]} boxes")

def main():
    """Run all tests"""
    print("Testing C++ implementation of scale_ray_nms_box3d")
    print("=" * 60)
    
    try:
        tests_passed = 0
        total_tests = 3
        
        if test_basic_functionality():
            tests_passed += 1
        
        if test_individual_nms_functions():
            tests_passed += 1
            
        if test_edge_cases():
            tests_passed += 1
        
        performance_test()
        
        print(f"\n=== SUMMARY ===")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("🎉 All tests PASSED! C++ implementation is working correctly.")
            return True
        else:
            print("⚠️  Some tests FAILED. Please check the implementation.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)