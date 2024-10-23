# 指定CMake的最低版本
cmake_minimum_required(VERSION 3.10)

# 定义项目名称和语言
project(AIgorithm LANGUAGES CXX)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 查找所需要的包
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
# 指定要编译的源文件
file(GLOB SOURCES "src/*.cpp")
add_executable(main ${SOURCES})    # 格式：(可执行文件 要编译的文件)

# 添加库到可执行文件
target_link_libraries(main ${OpenCV_LIBS} Threads::Threads nvonnxparser nvinfer rt ${CUDA_LIBRARIES}) 
# 添加头文件到可执行文件
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS}                          # opencv
                                        /home/jetson/Desktop/MyAlgorithm/include        # include
                                        /usr/include/x86_64-linux-gnu/                  # tensorrt
                                        /usr/local/cuda/include)                        # cuda
