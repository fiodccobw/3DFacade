cmake_minimum_required(VERSION 3.3)
get_filename_component(3dFacade ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(3dFacade)
include(./FindGUROBI.cmake)
include_directories(${GUROBI_INCLUDE_DIRS})

include_directories(
        ${GUROBI_INCLUDE_DIRS}
)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


find_package( OpenCV REQUIRED )
find_package(Easy3D REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)


include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Hide variables from CMake-Gui options
mark_as_advanced(GUROBI_LIBRARIES GUROBI_INCLUDE_DIRS GUROBI_INCLUDE_DIR)

add_executable(3dFacade
        main.cpp
        utils.h
        ../3rd_party/ETH3DFormatLoader/cameras.cc ../3rd_party/ETH3DFormatLoader/cameras.h ../3rd_party/ETH3DFormatLoader/images.cc ../3rd_party/ETH3DFormatLoader/images.h facadeModeling.h facadeModeling.cpp utils.cpp reconstruction.h reconstruction.cpp)

target_link_libraries(3dFacade  ${OpenCV_LIBS} ${GUROBI_LIBRARIES} easy3d::core easy3d::fileio easy3d::renderer easy3d::viewer Open3D::Open3D)