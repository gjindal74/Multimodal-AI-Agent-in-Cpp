cmake_minimum_required(VERSION 3.18)
project(multimodal_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PortAudio for audio capture
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
# llama.cpp for LLM reasoning
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH "llama.cpp directory")
find_library(LLAMA_LIB 
    NAMES llama libllama
    PATHS 
        ${LLAMA_DIR}/build/bin
    NO_DEFAULT_PATH
)

find_library(GGML_LIB 
    NAMES ggml libggml
    PATHS 
        ${LLAMA_DIR}/build/bin
    NO_DEFAULT_PATH
)

find_path(LLAMA_INCLUDE llama.h PATHS ${LLAMA_DIR}/include NO_DEFAULT_PATH)

if(NOT LLAMA_LIB)
    message(FATAL_ERROR "llama library not found.")
endif()

message(STATUS "Found llama library: ${LLAMA_LIB}")
message(STATUS "Found ggml library: ${GGML_LIB}")
message(STATUS "llama include directory: ${LLAMA_INCLUDE}")
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