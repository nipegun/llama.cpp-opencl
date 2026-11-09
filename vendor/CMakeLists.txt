# used by common, the tools and the tests
add_subdirectory(nlohmann)
add_subdirectory(sheredom)

# only used by common
if (LLAMA_BUILD_COMMON)
    add_subdirectory(cpp-httplib)
endif()
