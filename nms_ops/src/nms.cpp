#include <vector>
#include <algorithm>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Intersection over Union calculation for 2D boxes
float iou_2d(const std::vector<float>& box1, const std::vector<float>& box2) {
    // box format: [x1, y1, x2, y2]
    float x1_inter = std::max(box1[0], box2[0]);
    float y1_inter = std::max(box1[1], box2[1]);
    float x2_inter = std::min(box1[2], box2[2]);
    float y2_inter = std::min(box1[3], box2[3]);
    
    if (x2_inter <= x1_inter || y2_inter <= y1_inter) {
        return 0.0f;
    }
    
    float inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter);
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - inter_area;
    
    return inter_area / union_area;
}

// Intersection over Union calculation for 3D boxes
float iou_3d(const std::vector<float>& box1, const std::vector<float>& box2) {
    // box format: [x1, y1, z1, x2, y2, z2]
    float x1_inter = std::max(box1[0], box2[0]);
    float y1_inter = std::max(box1[1], box2[1]);
    float z1_inter = std::max(box1[2], box2[2]);
    float x2_inter = std::min(box1[3], box2[3]);
    float y2_inter = std::min(box1[4], box2[4]);
    float z2_inter = std::min(box1[5], box2[5]);
    
    if (x2_inter <= x1_inter || y2_inter <= y1_inter || z2_inter <= z1_inter) {
        return 0.0f;
    }
    
    float inter_volume = (x2_inter - x1_inter) * (y2_inter - y1_inter) * (z2_inter - z1_inter);
    float volume1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2]);
    float volume2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2]);
    float union_volume = volume1 + volume2 - inter_volume;
    
    return inter_volume / union_volume;
}

// Bird's Eye View NMS implementation
std::vector<int> bev_nms(py::array_t<float> boxes, py::array_t<float> scores, float iou_threshold) {
    auto boxes_buf = boxes.request();
    auto scores_buf = scores.request();
    
    if (boxes_buf.ndim != 2 || boxes_buf.shape[1] != 4) {
        throw std::runtime_error("Boxes should be N x 4 array");
    }
    if (scores_buf.ndim != 1) {
        throw std::runtime_error("Scores should be 1D array");
    }
    
    int num_boxes = boxes_buf.shape[0];
    float* boxes_ptr = static_cast<float*>(boxes_buf.ptr);
    float* scores_ptr = static_cast<float*>(scores_buf.ptr);
    
    // Create indices and sort by scores in descending order
    std::vector<int> indices(num_boxes);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return scores_ptr[i] > scores_ptr[j];
    });
    
    std::vector<bool> suppressed(num_boxes, false);
    std::vector<int> keep;
    
    for (int i = 0; i < num_boxes; ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        std::vector<float> box1(boxes_ptr + idx * 4, boxes_ptr + (idx + 1) * 4);
        
        for (int j = i + 1; j < num_boxes; ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            std::vector<float> box2(boxes_ptr + idx2 * 4, boxes_ptr + (idx2 + 1) * 4);
            
            if (iou_2d(box1, box2) > iou_threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}

// Ray-based NMS implementation
std::vector<int> ray_nms(py::array_t<float> boxes, py::array_t<float> scores, float iou_threshold) {
    auto boxes_buf = boxes.request();
    auto scores_buf = scores.request();
    
    if (boxes_buf.ndim != 2 || boxes_buf.shape[1] != 6) {
        throw std::runtime_error("Boxes should be N x 6 array for 3D boxes");
    }
    if (scores_buf.ndim != 1) {
        throw std::runtime_error("Scores should be 1D array");
    }
    
    int num_boxes = boxes_buf.shape[0];
    float* boxes_ptr = static_cast<float*>(boxes_buf.ptr);
    float* scores_ptr = static_cast<float*>(scores_buf.ptr);
    
    // Create indices and sort by scores in descending order
    std::vector<int> indices(num_boxes);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return scores_ptr[i] > scores_ptr[j];
    });
    
    std::vector<bool> suppressed(num_boxes, false);
    std::vector<int> keep;
    
    for (int i = 0; i < num_boxes; ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        std::vector<float> box1(boxes_ptr + idx * 6, boxes_ptr + (idx + 1) * 6);
        
        for (int j = i + 1; j < num_boxes; ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            std::vector<float> box2(boxes_ptr + idx2 * 6, boxes_ptr + (idx2 + 1) * 6);
            
            if (iou_3d(box1, box2) > iou_threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}

// Scale Ray NMS implementation - combines multi-scale processing with ray-based NMS
std::vector<int> scale_ray_nms(py::array_t<float> boxes, py::array_t<float> scores, 
                               py::array_t<float> scales, float iou_threshold, 
                               float scale_threshold = 0.1) {
    auto boxes_buf = boxes.request();
    auto scores_buf = scores.request();
    auto scales_buf = scales.request();
    
    if (boxes_buf.ndim != 2 || boxes_buf.shape[1] != 6) {
        throw std::runtime_error("Boxes should be N x 6 array for 3D boxes");
    }
    if (scores_buf.ndim != 1) {
        throw std::runtime_error("Scores should be 1D array");
    }
    if (scales_buf.ndim != 1) {
        throw std::runtime_error("Scales should be 1D array");
    }
    
    int num_boxes = boxes_buf.shape[0];
    float* boxes_ptr = static_cast<float*>(boxes_buf.ptr);
    float* scores_ptr = static_cast<float*>(scores_buf.ptr);
    float* scales_ptr = static_cast<float*>(scales_buf.ptr);
    
    // Create indices and sort by scores in descending order
    std::vector<int> indices(num_boxes);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return scores_ptr[i] > scores_ptr[j];
    });
    
    std::vector<bool> suppressed(num_boxes, false);
    std::vector<int> keep;
    
    for (int i = 0; i < num_boxes; ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        std::vector<float> box1(boxes_ptr + idx * 6, boxes_ptr + (idx + 1) * 6);
        float scale1 = scales_ptr[idx];
        
        for (int j = i + 1; j < num_boxes; ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            std::vector<float> box2(boxes_ptr + idx2 * 6, boxes_ptr + (idx2 + 1) * 6);
            float scale2 = scales_ptr[idx2];
            
            // Check scale similarity - only apply NMS if scales are similar
            float scale_ratio = std::abs(scale1 - scale2) / std::max(scale1, scale2);
            if (scale_ratio > scale_threshold) {
                continue; // Skip NMS for boxes with very different scales
            }
            
            // Apply ray-based IoU calculation with scale consideration
            float base_iou = iou_3d(box1, box2);
            
            // Adjust IoU threshold based on scale difference
            float adjusted_threshold = iou_threshold * (1.0f - scale_ratio * 0.5f);
            
            if (base_iou > adjusted_threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}