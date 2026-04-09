cmake_minimum_required(VERSION 3.16)
project(ternary_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ternary_infer
    src/main.cpp
    src/model.cpp
    src/tensor.cpp
    src/ternary_kernel.cpp
    src/layers.cpp
    src/resnet20.cpp
)

target_include_directories(ternary_infer PRIVATE include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ternary_infer OpenMP::OpenMP_CXX)
endif()

if (MSVC)
    target_compile_options(ternary_infer PRIVATE /O2)
else()
    target_compile_options(ternary_infer PRIVATE
        -O3
        -DNDEBUG
        -march=native
        -mavx2
        -mfma
        -mbmi2
        -funroll-loops
        -fno-math-errno
        -fomit-frame-pointer
        -ffp-contract=fast
    )
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if (ipo_supported)
        set_property(TARGET ternary_infer PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()
