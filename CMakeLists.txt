# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 3.14)
# 声明一个 cmake 工程
project(ocr)
# 指定为C++11 版本
set(CMAKE_CXX_STANDARD 11)
# 或命令行指定-DCMAKE_CXX_FLAGS=-g
set(CMAKE_CXX_FLAGS "-g -o3 -pthread -fopenmp")

# 设置依赖库
set(depends)

# paddle ocr
SET(PADDLE_LIB "/home/ssj/software/paddle_inference" CACHE PATH "Location of libraries")
include_directories("${PADDLE_LIB}/paddle/include")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/install/onnxruntime/include")
include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/paddle/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/onnxruntime/lib")
link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")            
set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)

set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS} ${MATH_LIB} ${MKLDNN_LIB} glog gflags protobuf z xxhash)
if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
  set(DEPS ${DEPS} snappystream)
endif()
if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
  set(DEPS ${DEPS} snappy)
endif()

# ppocr
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/pp-ocr/include) 
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/pp-ocr/lib)
list(APPEND depends ppocr ${DEPS})

# cuda
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
list(APPEND depends cudart)

# opencv
set(OpenCV_DIR /home/dell/software/opencv-4.5.2/build)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND depends ${OpenCV_LIBS})

# yaml-cpp
# include_directories(${PROJECT_SOURCE_DIR}/3rdparty/yaml-cpp/include) 
# link_directories(${PROJECT_SOURCE_DIR}/3rdparty/yaml-cpp/lib)
list(APPEND depends yaml-cpp)

# 添加包含目录
include_directories(${PROJECT_SOURCE_DIR}/include) # "路径1" "路径2"

# 编译成一个可执行程序
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRCS)
add_executable(ocr ${SRCS})
target_link_libraries (ocr ${depends})