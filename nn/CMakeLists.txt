# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

include_directories(src/main/cpp/rga)

add_library(
        rknn4j
        SHARED
        #        加速so库头文件
        src/main/cpp/rknn_api.h
        #        加速实现部分
        src/main/cpp/blazeface/bz_rknn_jni.cc
        src/main/cpp/pfld/pfld_rknn_jni.cc

        #        工具类部分
        src/main/cpp/utils.h
        src/main/cpp/utils.cpp
)


find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

#为rknn连接so库
target_link_libraries( # Specifies the target library.
        rknn4j
        #
        #         Links the target library to the log library
        #         included in the NDK.
        ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/librknnrt.so
        ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/librga.so

        ${log-lib}
)
