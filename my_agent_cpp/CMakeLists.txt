cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(agent_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Change this to your actual LibTorch path
set(Torch_DIR "C:/libtorch-win-shared-with-deps-2.9.1+cpu/libtorch/share/cmake/Torch")   # Windows
# set(Torch_DIR "/home/you/libtorch/share/cmake/Torch")  # Linux/macOS

find_package(Torch REQUIRED)

add_executable(agent_inference main.cpp)
target_link_libraries(agent_inference "${TORCH_LIBRARIES}")
