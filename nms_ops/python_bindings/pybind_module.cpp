#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nms_ops.h"

namespace py = pybind11;

// Helper function to convert numpy array to vector of Box3D
std::vector<nms_ops::Box3D> numpy_to_boxes(py::array_t<float> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 2 || buf_info.shape[1] != 7) {
        throw std::runtime_error("Input array must be of shape (N, 7)");
    }
    
    int n_boxes = buf_info.shape[0];
    float* ptr = static_cast<float*>(buf_info.ptr);
    
    return nms_ops::array_to_boxes(ptr, n_boxes);
}

// Helper function to convert vector of Box3D to numpy array
py::array_t<float> boxes_to_numpy(const std::vector<nms_ops::Box3D>& boxes) {
    if (boxes.empty()) {
        return py::array_t<float>(py::array::ShapeContainer({0, 7}));
    }
    
    py::array_t<float> result = py::array_t<float>(
        py::array::ShapeContainer({static_cast<int>(boxes.size()), 7})
    );
    
    py::buffer_info buf_info = result.request();
    float* ptr = static_cast<float*>(buf_info.ptr);
    
    nms_ops::boxes_to_array(boxes, ptr);
    
    return result;
}

// Helper function to convert numpy array to vector of floats
std::vector<float> numpy_to_vector(py::array_t<float> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    
    float* ptr = static_cast<float*>(buf_info.ptr);
    return std::vector<float>(ptr, ptr + buf_info.shape[0]);
}

// Helper function to convert vector of floats to numpy array
py::array_t<float> vector_to_numpy(const std::vector<float>& vec) {
    if (vec.empty()) {
        return py::array_t<float>(py::array::ShapeContainer({0}));
    }
    
    py::array_t<float> result = py::array_t<float>(
        py::array::ShapeContainer({static_cast<int>(vec.size())})
    );
    
    py::buffer_info buf_info = result.request();
    float* ptr = static_cast<float*>(buf_info.ptr);
    
    std::copy(vec.begin(), vec.end(), ptr);
    
    return result;
}

// Python wrapper for bev_nms
py::array_t<int> py_bev_nms(py::array_t<float> boxes, 
                           py::array_t<float> scores, 
                           float thresh) {
    std::vector<nms_ops::Box3D> box_vec = numpy_to_boxes(boxes);
    std::vector<float> score_vec = numpy_to_vector(scores);
    
    std::vector<int> keep = nms_ops::bev_nms(box_vec, score_vec, thresh);
    
    if (keep.empty()) {
        return py::array_t<int>(py::array::ShapeContainer({0}));
    }
    
    py::array_t<int> result = py::array_t<int>(
        py::array::ShapeContainer({static_cast<int>(keep.size())})
    );
    
    py::buffer_info buf_info = result.request();
    int* ptr = static_cast<int*>(buf_info.ptr);
    
    std::copy(keep.begin(), keep.end(), ptr);
    
    return result;
}

// Python wrapper for ray_nms
py::array_t<int> py_ray_nms(py::array_t<float> boxes, 
                           py::array_t<float> scores, 
                           float thresh) {
    std::vector<nms_ops::Box3D> box_vec = numpy_to_boxes(boxes);
    std::vector<float> score_vec = numpy_to_vector(scores);
    
    std::vector<int> keep = nms_ops::ray_nms(box_vec, score_vec, thresh);
    
    if (keep.empty()) {
        return py::array_t<int>(py::array::ShapeContainer({0}));
    }
    
    py::array_t<int> result = py::array_t<int>(
        py::array::ShapeContainer({static_cast<int>(keep.size())})
    );
    
    py::buffer_info buf_info = result.request();
    int* ptr = static_cast<int*>(buf_info.ptr);
    
    std::copy(keep.begin(), keep.end(), ptr);
    
    return result;
}

// Python wrapper for scale_ray_nms_box3d
py::tuple py_scale_ray_nms_box3d(py::array_t<float> boxes,
                                py::array_t<float> scores,
                                float thresh_bev = 0.5f,
                                float thresh_ray = 0.3f,
                                float scale_factor = 1.0f) {
    std::vector<nms_ops::Box3D> box_vec = numpy_to_boxes(boxes);
    std::vector<float> score_vec = numpy_to_vector(scores);
    
    auto [result_boxes, result_scores] = nms_ops::scale_ray_nms_box3d(
        box_vec, score_vec, thresh_bev, thresh_ray, scale_factor
    );
    
    py::array_t<float> py_boxes = boxes_to_numpy(result_boxes);
    py::array_t<float> py_scores = vector_to_numpy(result_scores);
    
    return py::make_tuple(py_boxes, py_scores);
}

PYBIND11_MODULE(nms_ops_cpp, m) {
    m.doc() = "C++ implementation of NMS operations for 3D boxes";
    
    // Bind the Box3D structure
    py::class_<nms_ops::Box3D>(m, "Box3D")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, float, float>())
        .def_readwrite("x", &nms_ops::Box3D::x)
        .def_readwrite("y", &nms_ops::Box3D::y)
        .def_readwrite("z", &nms_ops::Box3D::z)
        .def_readwrite("l", &nms_ops::Box3D::l)
        .def_readwrite("w", &nms_ops::Box3D::w)
        .def_readwrite("h", &nms_ops::Box3D::h)
        .def_readwrite("ry", &nms_ops::Box3D::ry);
    
    // Bind the main functions
    m.def("bev_nms", &py_bev_nms, 
          "Bird's Eye View Non-Maximum Suppression",
          py::arg("boxes"), py::arg("scores"), py::arg("thresh"));
    
    m.def("ray_nms", &py_ray_nms,
          "Ray-based Non-Maximum Suppression for 3D boxes", 
          py::arg("boxes"), py::arg("scores"), py::arg("thresh"));
    
    m.def("scale_ray_nms_box3d", &py_scale_ray_nms_box3d,
          "Scale and apply ray-based NMS on 3D boxes",
          py::arg("boxes"), py::arg("scores"), 
          py::arg("thresh_bev") = 0.5f, py::arg("thresh_ray") = 0.3f, 
          py::arg("scale_factor") = 1.0f);
    
    // Version info
    m.attr("__version__") = "1.0.0";
}