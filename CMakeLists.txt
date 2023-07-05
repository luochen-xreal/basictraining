cmake_minimum_required(VERSION 2.8)
project(TASK)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -Wall  -O3 -march=native -fsanitize=address -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -Wall   -O3 -march=native -fsanitize=address -g")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()



find_package(Eigen3 REQUIRED NO_MODULE)
message("EIGEN3_INCLUDE_DIR:" ${EIGEN3_INCLUDE_DIR})
# message("EIGEN3_LIBS:" ${EIGEN3_LIBS})
find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
   endif()
endif()
message("OpenCV_INCLUDE_DIRS:" ${OpenCV_INCLUDE_DIRS})
# message("EIGEN3_LIBS:" ${EIGEN3_LIBS})
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
)



add_library(STATIC_TESTLIB STATIC
src/TestFunction.cc
)
add_library(TESTLIB SHARED
src/TestFunction.cc
src/ExtractAndMatch.cc
)
target_link_libraries(TESTLIB 
${OpenCV_LIBS}
)
add_executable(test 
test.cc
)
add_executable(eigentest
EigenTest.cc
)
add_executable(LinearSolver 
LinearSolver.cc
)
target_link_libraries(test 
TESTLIB
) 
add_executable(OpenCVTest
OpenCVTest.cc 
)
target_link_libraries(OpenCVTest
 TESTLIB 
)

add_executable(test1
test1.cc
)
target_link_libraries(test1
${OpenCV_LIBS}
)