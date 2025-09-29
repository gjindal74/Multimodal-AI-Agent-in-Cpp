find_package(OpenCV REQUIRED)

# Find PortAudio
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

add_executable(agent_app
    main.cpp
    vision/visionmodule.cpp
    audio/audio.cpp
)

target_include_directories(agent_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include
    ${PORTAUDIO_INCLUDE_DIRS}
    ${WHISPER_INCLUDE}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_link_directories(agent_app PRIVATE
    /opt/homebrew/lib
)

target_link_libraries(agent_app PRIVATE
    ${OpenCV_LIBS}
    /opt/homebrew/lib/libonnxruntime.dylib
    /opt/homebrew/lib/libportaudio.dylib  # Explicit path
    ${WHISPER_LIB}
)