cmake_minimum_required(VERSION 3.18)
project(multimodal_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PortAudio for audio capture
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Whisper.cpp for speech-to-text
set(WHISPER_DIR "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp" CACHE PATH "Whisper.cpp directory")

# Find whisper dynamic library
find_library(WHISPER_LIB 
    NAMES whisper libwhisper
    PATHS 
        ${WHISPER_DIR}/build/src
        ${WHISPER_DIR}/build
        ${WHISPER_DIR}
    NO_DEFAULT_PATH
)

find_path(WHISPER_INCLUDE whisper.h PATHS ${WHISPER_DIR}/include NO_DEFAULT_PATH)

if(NOT WHISPER_LIB)
    message(FATAL_ERROR "Whisper library not found. Searched in:\n  ${WHISPER_DIR}/build/src\n  ${WHISPER_DIR}/build\nPlease build whisper.cpp first.")
endif()

if(NOT WHISPER_INCLUDE)
    message(FATAL_ERROR "Whisper headers not found at ${WHISPER_DIR}/include")
endif()

message(STATUS "Found Whisper library: ${WHISPER_LIB}")
message(STATUS "Whisper include directory: ${WHISPER_INCLUDE}")

add_subdirectory(src)