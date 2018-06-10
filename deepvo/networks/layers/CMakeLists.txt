cmake_minimum_required(VERSION 2.8)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import sys, tensorflow; sys.stdout.write(tensorflow.sysconfig.get_include()); sys.stdout.flush()" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python -c "import sys, tensorflow; sys.stdout.write(tensorflow.sysconfig.get_lib()); sys.stdout.flush()" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)

message("Include libraries are: ${Tensorflow_INCLUDE_DIRS}")

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")

link_directories(${Tensorflow_LIB_DIRS})
link_libraries(-ltensorflow_framework)

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(gvnn SHARED se3toMatrixRt.cc se3toMatrixRt_grad.cc)
