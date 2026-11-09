# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    add_subdirectory(eval-callback)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    if (GGML_OPENCL)
        add_subdirectory(bench-dmmv)
        add_subdirectory(bench-gemm)
    endif()
endif()
