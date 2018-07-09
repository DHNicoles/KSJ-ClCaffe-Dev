
#caffe
set(ViennaCL_INCLUDE_DIR ../include/viennacl_headfile/)
include_directories(
    ${ViennaCL_INCLUDE_DIR}
)

find_library(CAFFE caffe ../lib)
set(Caffe_LIBRARIES ${CAFFE})
add_definitions(${Caffe_DEFINITIONS})
add_definitions(-DUSE_GREENTEA -DUSE_OPENCV -DHAS_HALF_SUPPORT)

#OpenCV
set(OpenCV_INCLUDE_DIRS ../include/opencv ../include/opencv2)
INCLUDE_DIRECTORIES({OpenCV_INCLUDE_DIRS})
find_library(HIGHGUI opencv_highgui ../lib)
find_library(CORE opencv_core ../lib)
find_library(IMAGE_PROC opencv_imgproc ../lib)
find_library(OBJDETECT opencv_objdetect ../lib)
find_library(VIDEO opencv_video ../lib)
set(OpenCV_LIBS ${HIGHGUI} ${CORE} ${IMAGE_PROC} ${OBJDETECT} ${VIDEO})

#HikVision [not used]
##find_library(HCNETSDK hcnetsdk ../lib)
##find_library(PLAYM4 PlayCtrl ../lib)
##find_library(AUDIO_RENDER AudioRender ../lib)
##find_library(SUPER_RENDER SuperRender ../lib)

#KSJ Vision
find_library(KSJAPI ksjapi ../lib)
find_library(KSJBAYER ksjbayer ../lib)
find_library(KSJLOG ksjlog ../lib)
set(KSJ_LIBS ${KSJAPI} ${KSJBAYER} ${KSJLOG})

#other
find_library(GLOG glog ../lib)
find_library(BOOST_SYS boost_system ../lib)
find_library(PTHREAD pthread ../lib)
find_library(OPENCL OpenCL ../lib)

#merge
SET(LIBS ${KSJ_LIBS} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${GLOG} ${PTHREAD} ${BOOST_SYS} ${OPENCL})

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -msse4.2 -fPIC -o3")
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(../include/atlas)
INCLUDE_DIRECTORIES(../include/cnn_multi_tracker)

file(GLOB_RECURSE SRC_LIST . "*.cpp")  
AUX_SOURCE_DIRECTORY(. SRC_LIST)

#LINK_DIRECTORIES(../../lib)

ADD_EXECUTABLE(headtracker ${SRC_LIST})

TARGET_LINK_LIBRARIES(headtracker ${LIBS})
