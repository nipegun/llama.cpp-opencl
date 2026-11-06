# OpenCL backend of llamacppOpenCL11: kernels written for OpenCL 1.1 devices (NVIDIA Fermi, AMD TeraScale,
# Intel Ivy Bridge with Beignet, ...). Only libOpenCL is needed.

find_package(OpenCL REQUIRED)

set(TARGET_NAME ggml-opencl)

ggml_add_backend_library(${TARGET_NAME}
                         ggml-opencl.cpp
                         ../../include/ggml-opencl.h)

target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
