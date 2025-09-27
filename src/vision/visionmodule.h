#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <vector>

struct Detection {
    std::string label;
    float score;
    cv::Rect box;
};

class VisionModule {
public:
    VisionModule(const std::string& modelPath);
    ~VisionModule(); // Added destructor for proper cleanup
    
    bool init();
    std::vector<Detection> detect(const cv::Mat& frame);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

private:
    std::string modelPath_;
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions sessionOptions_;
    std::vector<const char*> inputNodeNames_;
    std::vector<const char*> outputNodeNames_;
    
    cv::Mat preprocess(const cv::Mat& frame);
};