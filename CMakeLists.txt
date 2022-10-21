cmake_minimum_required(VERSION 3.0.0)
project(EyeInHand_Calibrate VERSION 0.1.0)

if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /source-charset:utf-8 /execution-charset:gbk")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8 /execution-charset:gbk")
endif()

#指定.h头文件目录列表
set(Include_DIR 
    ./Eigen 
    ./include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

#添加.h头文件目录到项目
include_directories(${Include_DIR})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})   # 包含CmakeList.txt所在目录

find_package(OpenCV
        PATHS D:\\OpenCV-4.5.0-withContribNonfree-VS2022-VScode # 指定第三方Opencv库的OpenCVConfig.cmake所在路径
        NO_DEFAULT_PATH
        REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS}) # 包含Opencv路径
    message( ${OpenCV_LIBS})    # 打印Opencv库Lib模块内容
else()
    message("OpenCV not found, so we won't build the project.")
endif()

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}) # 指定.exe文件输出路径为CmakeList.txt所在目录


#指定.cpp源文件列表并存于变量 DIR_SRCS
file(GLOB  SRCS_LIST
    ./*.cpp
    ./src/*.cpp
)
add_executable(${PROJECT_NAME} ${SRCS_LIST}) # 基于指定的源文件，创建可执行文件
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ) # 链接 目标和其依赖项所要使用的库。

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
