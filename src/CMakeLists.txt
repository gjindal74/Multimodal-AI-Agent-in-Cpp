find_package(OpenCV REQUIRED)

add_executable(agent_app
    main.cpp
    vision/visionmodule.cpp
    audio/audio.cpp
    llm/llmmodule.cpp
)

target_include_directories(agent_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include
    ${WHISPER_INCLUDE}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${LLAMA_INCLUDE}
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

# Set rpath to prioritize llama.cpp's libraries
set_target_properties(agent_app PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin:${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/build"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin:${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/build"
)

target_link_libraries(agent_app PRIVATE
    ${OpenCV_LIBS}
    /opt/homebrew/lib/libonnxruntime.dylib
    /opt/homebrew/lib/libportaudio.dylib
    
    # Whisper and its ggml
    ${WHISPER_LIB}
    
    # Llama.cpp and its ggml (link AFTER whisper to take precedence)
    ${LLAMA_LIB}
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin/libggml-base.dylib
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin/libggml-metal.dylib
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/build/bin/libggml-cpu.dylib
)