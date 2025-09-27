find_package(OpenCV REQUIRED)

add_executable(agent_app
    main.cpp
    vision/visionmodule.cpp
)

target_include_directories(agent_app PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include  # Use the parent include folder
)

target_link_libraries(agent_app PRIVATE
    ${OpenCV_LIBS}
    /opt/homebrew/lib/libonnxruntime.dylib
)
