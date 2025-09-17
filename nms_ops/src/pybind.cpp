#include "nms.cpp"

PYBIND11_MODULE(nms_ops, m) {
    m.doc() = "NMS operations for object detection";
    
    m.def("bev_nms", &bev_nms, 
          "Bird's Eye View Non-Maximum Suppression",
          py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"));
    
    m.def("ray_nms", &ray_nms, 
          "Ray-based Non-Maximum Suppression for 3D boxes",
          py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"));
    
    m.def("scale_ray_nms", &scale_ray_nms, 
          "Scale-aware Ray-based Non-Maximum Suppression",
          py::arg("boxes"), py::arg("scores"), py::arg("scales"), 
          py::arg("iou_threshold"), py::arg("scale_threshold") = 0.1);
}