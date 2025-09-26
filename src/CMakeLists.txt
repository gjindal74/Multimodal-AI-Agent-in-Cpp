# Find dependencies first
find_package(OpenCV REQUIRED)

# Add executable
add_executable(agent_app main.cpp)

# Link OpenCV
target_include_directories(agent_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(agent_app PRIVATE ${OpenCV_LIBS})

