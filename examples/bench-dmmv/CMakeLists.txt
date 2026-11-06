# bench-dmmv: micro-benchmark of the OpenCL backend kernels (OpenCL builds only). It #includes ggml-opencl.cpp as a
# unity translation unit, so it is linked against ggml-base only (CPU reference: quantization and to_float).

set(TARGET llama-bench-dmmv)
add_executable(${TARGET} bench-dmmv.cpp)
find_package(OpenCL REQUIRED)
find_package(OpenMP)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-opencl)
target_link_libraries(${TARGET} PRIVATE ggml-base OpenCL::OpenCL ${CMAKE_THREAD_LIBS_INIT})
if (OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
