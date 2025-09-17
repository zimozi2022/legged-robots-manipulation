#include "nms_ops.h"
#include <algorithm>
#include <numeric>

namespace nms_ops {

std::vector<int> bev_nms(const std::vector<Box3D>& boxes, 
                        const std::vector<float>& scores, 
                        float thresh) {
    if (boxes.empty()) {
        return {};
    }
    
    int n = boxes.size();
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    
    // Sort by scores in descending order
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);
    
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        // Calculate BEV coordinates for current box
        float x1_i = boxes[idx].x - boxes[idx].l / 2.0f;
        float y1_i = boxes[idx].y - boxes[idx].w / 2.0f;
        float x2_i = boxes[idx].x + boxes[idx].l / 2.0f;
        float y2_i = boxes[idx].y + boxes[idx].w / 2.0f;
        float area_i = boxes[idx].l * boxes[idx].w;
        
        // Suppress overlapping boxes
        for (int j = i + 1; j < n; ++j) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            
            // Calculate BEV coordinates for comparison box
            float x1_j = boxes[jdx].x - boxes[jdx].l / 2.0f;
            float y1_j = boxes[jdx].y - boxes[jdx].w / 2.0f;
            float x2_j = boxes[jdx].x + boxes[jdx].l / 2.0f;
            float y2_j = boxes[jdx].y + boxes[jdx].w / 2.0f;
            float area_j = boxes[jdx].l * boxes[jdx].w;
            
            // Calculate intersection
            float xx1 = std::max(x1_i, x1_j);
            float yy1 = std::max(y1_i, y1_j);
            float xx2 = std::min(x2_i, x2_j);
            float yy2 = std::min(y2_i, y2_j);
            
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;
            
            float iou = inter / (area_i + area_j - inter);
            
            if (iou > thresh) {
                suppressed[jdx] = true;
            }
        }
    }
    
    return keep;
}

std::vector<int> ray_nms(const std::vector<Box3D>& boxes, 
                        const std::vector<float>& scores, 
                        float thresh) {
    if (boxes.empty()) {
        return {};
    }
    
    int n = boxes.size();
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    
    // Sort by scores in descending order
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);
    
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        // Calculate 3D coordinates for current box
        float x1_i = boxes[idx].x - boxes[idx].l / 2.0f;
        float y1_i = boxes[idx].y - boxes[idx].w / 2.0f;
        float z1_i = boxes[idx].z - boxes[idx].h / 2.0f;
        float x2_i = boxes[idx].x + boxes[idx].l / 2.0f;
        float y2_i = boxes[idx].y + boxes[idx].w / 2.0f;
        float z2_i = boxes[idx].z + boxes[idx].h / 2.0f;
        float volume_i = boxes[idx].l * boxes[idx].w * boxes[idx].h;
        
        // Suppress overlapping boxes
        for (int j = i + 1; j < n; ++j) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            
            // Calculate 3D coordinates for comparison box
            float x1_j = boxes[jdx].x - boxes[jdx].l / 2.0f;
            float y1_j = boxes[jdx].y - boxes[jdx].w / 2.0f;
            float z1_j = boxes[jdx].z - boxes[jdx].h / 2.0f;
            float x2_j = boxes[jdx].x + boxes[jdx].l / 2.0f;
            float y2_j = boxes[jdx].y + boxes[jdx].w / 2.0f;
            float z2_j = boxes[jdx].z + boxes[jdx].h / 2.0f;
            float volume_j = boxes[jdx].l * boxes[jdx].w * boxes[jdx].h;
            
            // Calculate 3D intersection
            float xx1 = std::max(x1_i, x1_j);
            float yy1 = std::max(y1_i, y1_j);
            float zz1 = std::max(z1_i, z1_j);
            float xx2 = std::min(x2_i, x2_j);
            float yy2 = std::min(y2_i, y2_j);
            float zz2 = std::min(z2_i, z2_j);
            
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float d = std::max(0.0f, zz2 - zz1);
            float inter = w * h * d;
            
            float iou = inter / (volume_i + volume_j - inter);
            
            if (iou > thresh) {
                suppressed[jdx] = true;
            }
        }
    }
    
    return keep;
}

std::tuple<std::vector<Box3D>, std::vector<float>> 
scale_ray_nms_box3d(const std::vector<Box3D>& boxes,
                    const std::vector<float>& scores,
                    float thresh_bev,
                    float thresh_ray,
                    float scale_factor) {
    if (boxes.empty()) {
        return std::make_tuple(std::vector<Box3D>(), std::vector<float>());
    }
    
    // Scale the boxes
    std::vector<Box3D> scaled_boxes = boxes;
    for (auto& box : scaled_boxes) {
        box.l *= scale_factor;
        box.w *= scale_factor;
        box.h *= scale_factor;
    }
    
    // Apply BEV NMS first
    std::vector<int> bev_keep = bev_nms(scaled_boxes, scores, thresh_bev);
    
    if (bev_keep.empty()) {
        return std::make_tuple(std::vector<Box3D>(), std::vector<float>());
    }
    
    // Extract boxes and scores that passed BEV NMS
    std::vector<Box3D> bev_boxes;
    std::vector<float> bev_scores;
    bev_boxes.reserve(bev_keep.size());
    bev_scores.reserve(bev_keep.size());
    
    for (int idx : bev_keep) {
        bev_boxes.push_back(scaled_boxes[idx]);
        bev_scores.push_back(scores[idx]);
    }
    
    // Apply Ray NMS on BEV results
    std::vector<int> ray_keep = ray_nms(bev_boxes, bev_scores, thresh_ray);
    
    // Map back to original indices and extract final results
    std::vector<Box3D> final_boxes;
    std::vector<float> final_scores;
    final_boxes.reserve(ray_keep.size());
    final_scores.reserve(ray_keep.size());
    
    for (int i : ray_keep) {
        int original_idx = bev_keep[i];
        final_boxes.push_back(boxes[original_idx]);  // Use original unscaled boxes
        final_scores.push_back(scores[original_idx]);
    }
    
    return std::make_tuple(final_boxes, final_scores);
}

std::vector<Box3D> array_to_boxes(const float* data, int n_boxes) {
    std::vector<Box3D> boxes;
    boxes.reserve(n_boxes);
    
    for (int i = 0; i < n_boxes; ++i) {
        const float* box_data = data + i * 7;
        boxes.emplace_back(box_data[0], box_data[1], box_data[2], 
                          box_data[3], box_data[4], box_data[5], box_data[6]);
    }
    
    return boxes;
}

void boxes_to_array(const std::vector<Box3D>& boxes, float* output) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        const Box3D& box = boxes[i];
        float* box_output = output + i * 7;
        box_output[0] = box.x;
        box_output[1] = box.y;
        box_output[2] = box.z;
        box_output[3] = box.l;
        box_output[4] = box.w;
        box_output[5] = box.h;
        box_output[6] = box.ry;
    }
}

} // namespace nms_ops