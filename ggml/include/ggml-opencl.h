#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// OpenCL backend of llamacppOpenCL11 (OpenCL 1.1 or newer, see ggml-opencl.cpp)

// registry with one device per OpenCL GPU found (or the devices selected with GGML_OPENCL_PLATFORM and
// GGML_OPENCL_DEVICE); only one of them can be used per process
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

// backend and buffer type of the first device of the registry
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);

GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
