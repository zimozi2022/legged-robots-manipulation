#pragma once

#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>

namespace nms_ops {

/**
 * @brief Structure to represent a 3D bounding box
 */
struct Box3D {
    float x, y, z;      // center coordinates
    float l, w, h;      // length, width, height (dimensions)
    float ry;           // rotation around y-axis
    
    Box3D() : x(0), y(0), z(0), l(0), w(0), h(0), ry(0) {}
    Box3D(float x_, float y_, float z_, float l_, float w_, float h_, float ry_)
        : x(x_), y(y_), z(z_), l(l_), w(w_), h(h_), ry(ry_) {}
};

/**
 * @brief Bird's Eye View Non-Maximum Suppression
 * 
 * @param boxes Vector of 3D boxes
 * @param scores Vector of confidence scores
 * @param thresh IoU threshold for suppression
 * @return Vector of indices of boxes to keep
 */
std::vector<int> bev_nms(const std::vector<Box3D>& boxes, 
                        const std::vector<float>& scores, 
                        float thresh);

/**
 * @brief Ray-based Non-Maximum Suppression for 3D boxes
 * 
 * @param boxes Vector of 3D boxes
 * @param scores Vector of confidence scores
 * @param thresh IoU threshold for suppression
 * @return Vector of indices of boxes to keep
 */
std::vector<int> ray_nms(const std::vector<Box3D>& boxes, 
                        const std::vector<float>& scores, 
                        float thresh);

/**
 * @brief Scale and apply ray-based NMS on 3D boxes
 * 
 * This function first scales the boxes, then applies BEV NMS followed by Ray NMS
 * to filter overlapping 3D bounding boxes.
 * 
 * @param boxes Vector of 3D boxes
 * @param scores Vector of confidence scores
 * @param thresh_bev IoU threshold for BEV NMS
 * @param thresh_ray IoU threshold for Ray NMS
 * @param scale_factor Factor to scale box dimensions
 * @return Tuple of (filtered_boxes, filtered_scores) after NMS
 */
std::tuple<std::vector<Box3D>, std::vector<float>> 
scale_ray_nms_box3d(const std::vector<Box3D>& boxes,
                    const std::vector<float>& scores,
                    float thresh_bev = 0.5f,
                    float thresh_ray = 0.3f,
                    float scale_factor = 1.0f);

/**
 * @brief Convert numpy-style array to vector of Box3D
 * 
 * @param data Flattened array data (N*7 elements)
 * @param n_boxes Number of boxes
 * @return Vector of Box3D objects
 */
std::vector<Box3D> array_to_boxes(const float* data, int n_boxes);

/**
 * @brief Convert vector of Box3D to numpy-style array
 * 
 * @param boxes Vector of Box3D objects
 * @param output Output array (must be pre-allocated with size boxes.size()*7)
 */
void boxes_to_array(const std::vector<Box3D>& boxes, float* output);

} // namespace nms_ops