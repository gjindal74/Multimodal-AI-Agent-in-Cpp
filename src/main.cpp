#include "vision/visionmodule.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>

// Simple tracking structure
struct TrackedObject {
    cv::Rect box;
    std::string label;
    float confidence;
    int missedFrames;
    int id;
};

class SimpleTracker {
private:
    std::map<int, TrackedObject> trackedObjects;
    int nextId = 0;
    const float IOU_THRESHOLD = 0.3f;
    const int MAX_MISSED_FRAMES = 5;
    
    float calculateIOU(const cv::Rect& a, const cv::Rect& b) {
        float inter = (a & b).area();
        float uni = a.area() + b.area() - inter;
        return (uni > 0) ? inter / uni : 0.0f;
    }
    
public:
    std::vector<Detection> updateTracks(const std::vector<Detection>& newDetections) {
        std::vector<Detection> smoothedDetections;
        std::vector<bool> matched(newDetections.size(), false);
        
        // Update existing tracks
        for (auto& [id, track] : trackedObjects) {
            track.missedFrames++;
            
            // Try to match with new detections
            float bestIOU = 0;
            int bestMatch = -1;
            
            for (size_t i = 0; i < newDetections.size(); i++) {
                if (matched[i] || newDetections[i].label != track.label) continue;
                
                float iou = calculateIOU(track.box, newDetections[i].box);
                if (iou > IOU_THRESHOLD && iou > bestIOU) {
                    bestIOU = iou;
                    bestMatch = static_cast<int>(i);
                }
            }
            
            if (bestMatch >= 0) {
                // Smooth the bounding box (simple averaging)
                const auto& det = newDetections[bestMatch];
                track.box.x = static_cast<int>(0.7f * track.box.x + 0.3f * det.box.x);
                track.box.y = static_cast<int>(0.7f * track.box.y + 0.3f * det.box.y);
                track.box.width = static_cast<int>(0.7f * track.box.width + 0.3f * det.box.width);
                track.box.height = static_cast<int>(0.7f * track.box.height + 0.3f * det.box.height);
                
                track.confidence = det.score;
                track.missedFrames = 0;
                matched[bestMatch] = true;
                
                smoothedDetections.push_back({track.label, track.confidence, track.box});
            }
        }
        
        // Remove lost tracks
        auto it = trackedObjects.begin();
        while (it != trackedObjects.end()) {
            if (it->second.missedFrames > MAX_MISSED_FRAMES) {
                it = trackedObjects.erase(it);
            } else {
                ++it;
            }
        }
        
        // Add new tracks for unmatched detections
        for (size_t i = 0; i < newDetections.size(); i++) {
            if (!matched[i]) {
                TrackedObject newTrack;
                newTrack.box = newDetections[i].box;
                newTrack.label = newDetections[i].label;
                newTrack.confidence = newDetections[i].score;
                newTrack.missedFrames = 0;
                newTrack.id = nextId++;
                
                trackedObjects[newTrack.id] = newTrack;
                smoothedDetections.push_back(newDetections[i]);
            }
        }
        
        return smoothedDetections;
    }
};

int main() {
    VisionModule vision("/Users/gaurangjindal/Desktop/multimodal-agent-cpp/models/yolov8n.onnx");

    if (!vision.init()) {
        std::cerr << "Failed to initialize vision module" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    
    // Set camera properties for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    SimpleTracker tracker;
    cv::Mat frame;
    
    // Performance monitoring
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float avgFPS = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Run detection
        auto detections = vision.detect(frame);
        
        // Apply tracking for smoother results
        auto smoothedDetections = tracker.updateTracks(detections);
        
        // Draw detections
        vision.drawDetections(frame, smoothedDetections);
        
        // Calculate and display FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
        
        if (duration.count() > 1000) { // Update every second
            avgFPS = frameCount * 1000.0f / duration.count();
            frameCount = 0;
            lastTime = currentTime;
        }
        
        // Display FPS on frame
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(avgFPS));
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Display detection count
        std::string countText = "Objects: " + std::to_string(smoothedDetections.size());
        cv::putText(frame, countText, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Enhanced Detections", frame);
        
        char key = cv::waitKey(1);
        if (key == 27) break; // ESC
        if (key == 's') { // Save screenshot
            cv::imwrite("detection_screenshot.jpg", frame);
            std::cout << "Screenshot saved!" << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}