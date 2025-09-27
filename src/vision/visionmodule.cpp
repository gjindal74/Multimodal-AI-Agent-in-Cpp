#include "visionmodule.h"
#include "coco_labels.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>

static constexpr int INPUT_WIDTH = 640;
static constexpr int INPUT_HEIGHT = 640;
static constexpr float CONF_THRESH = 0.25f;  // Lowered for better detection of non-person objects
static constexpr float NMS_THRESH = 0.4f;

struct Candidate {
    cv::Rect box;
    float score;
    int classId;
};

static float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return (uni > 0) ? inter / uni : 0.0f;
}

VisionModule::VisionModule(const std::string& modelPath)
    : modelPath_(modelPath), env_(ORT_LOGGING_LEVEL_WARNING, "Vision") {
    session_ = nullptr;
}

VisionModule::~VisionModule() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
    
    // Free allocated input/output names
    for (auto name : inputNodeNames_) {
        if (name) free(const_cast<char*>(name));
    }
    for (auto name : outputNodeNames_) {
        if (name) free(const_cast<char*>(name));
    }
}

bool VisionModule::init() {
    try {
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = new Ort::Session(env_, modelPath_.c_str(), sessionOptions_);

        size_t numInputNodes = session_->GetInputCount();
        inputNodeNames_.resize(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++) {
            auto name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            inputNodeNames_[i] = strdup(name.get());
        }

        size_t numOutputNodes = session_->GetOutputCount();
        outputNodeNames_.resize(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            outputNodeNames_[i] = strdup(name.get());
        }
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime init failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat VisionModule::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    return resized;
}

std::vector<Detection> VisionModule::detect(const cv::Mat& frame) {
    std::vector<Detection> results;
    
    if (frame.empty()) {
        std::cerr << "Empty frame provided to detect()" << std::endl;
        return results;
    }
    
    cv::Mat blob = preprocess(frame);
    
    // Convert HWC → CHW
    std::vector<float> inputTensorValues(INPUT_HEIGHT * INPUT_WIDTH * 3);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < INPUT_HEIGHT; y++) {
            for (int x = 0; x < INPUT_WIDTH; x++) {
                inputTensorValues[c * INPUT_HEIGHT * INPUT_WIDTH + y * INPUT_WIDTH + x] =
                    blob.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    std::vector<int64_t> inputDims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(),
        inputTensorValues.size(), inputDims.data(), inputDims.size());

    try {
        // Run inference
        auto output = session_->Run(Ort::RunOptions{nullptr},
            inputNodeNames_.data(), &inputTensor, 1,
            outputNodeNames_.data(), outputNodeNames_.size());

        // YOLOv8 output: [1, 84, 8400] → [batch, num_attrs, num_preds]
        float* data = output[0].GetTensorMutableData<float>();
        auto shape = output[0].GetTensorTypeAndShapeInfo().GetShape();

        // Debug output (remove in production)
        std::cout << "YOLO Output Shape: [";
        for (auto s : shape) std::cout << s << " ";
        std::cout << "]" << std::endl;

        int64_t batch = shape[0];
        int64_t numAttrs = shape[1]; // 84
        int64_t numPreds = shape[2]; // 8400

        std::vector<Candidate> candidates;
        
        // Calculate scale factors for coordinate conversion
        float scaleX = static_cast<float>(frame.cols) / INPUT_WIDTH;
        float scaleY = static_cast<float>(frame.rows) / INPUT_HEIGHT;

        for (int64_t i = 0; i < numPreds; i++) {
            float x = data[0 * numPreds + i];        // x center (0-640)
            float y = data[1 * numPreds + i];        // y center (0-640)  
            float w = data[2 * numPreds + i];        // width (0-640)
            float h = data[3 * numPreds + i];        // height (0-640)

            // Find class with highest score
            float maxScore = 0;
            int classId = -1;
            for (int64_t c = 4; c < numAttrs; c++) {
                float score = data[c * numPreds + i];
                if (score > maxScore) {
                    maxScore = score;
                    classId = static_cast<int>(c - 4);
                }
            }

            if (maxScore > CONF_THRESH) {
                // Use class-specific confidence thresholds
                float classThreshold = CONF_THRESH;
                
                // Person class (0) - higher threshold since it works well
                if (classId == 0) classThreshold = 0.5f;
                // Common objects - lower thresholds
                else if (classId == 56 || classId == 57) classThreshold = 0.2f; // chair, couch
                else if (classId == 59 || classId == 60) classThreshold = 0.25f; // dining table, toilet  
                else if (classId == 61 || classId == 62 || classId == 63) classThreshold = 0.2f; // tv, laptop, mouse
                else if (classId == 64 || classId == 65) classThreshold = 0.25f; // remote, keyboard
                else if (classId == 66) classThreshold = 0.2f; // cell phone
                else if (classId == 67 || classId == 68) classThreshold = 0.25f; // microwave, oven
                else if (classId == 73 || classId == 74) classThreshold = 0.2f; // book, clock
                // Vehicles and larger objects
                else if (classId >= 1 && classId <= 9) classThreshold = 0.3f; // vehicles
                // Animals
                else if (classId >= 14 && classId <= 23) classThreshold = 0.3f; // animals
                // Other objects
                else classThreshold = 0.25f;
                
                if (maxScore > classThreshold) {
                    // Convert from model coordinates (0-640) to original frame coordinates
                    float centerX = x * scaleX;
                    float centerY = y * scaleY;
                    float boxWidth = w * scaleX;
                    float boxHeight = h * scaleY;
                    
                    int left = static_cast<int>(centerX - boxWidth / 2.0f);
                    int top = static_cast<int>(centerY - boxHeight / 2.0f);
                    int width = static_cast<int>(boxWidth);
                    int height = static_cast<int>(boxHeight);

                    // Clamp to frame boundaries and validate
                    left = std::max(0, left);
                    top = std::max(0, top);
                    width = std::min(width, frame.cols - left);
                    height = std::min(height, frame.rows - top);

                    // Additional validation: reject boxes that are too small or too large
                    float boxArea = static_cast<float>(width * height);
                    float frameArea = static_cast<float>(frame.cols * frame.rows);
                    float areaRatio = boxArea / frameArea;
                    
                    // Class-specific area validation
                    float minAreaRatio = 0.0005f; // Very small objects allowed
                    float maxAreaRatio = 0.95f;   // Allow larger objects
                    
                    // Person class - stricter area requirements
                    if (classId == 0) {
                        minAreaRatio = 0.01f;  // Persons should be reasonably sized
                        maxAreaRatio = 0.8f;
                    }
                    // Small objects like phones, remotes, etc.
                    else if (classId == 64 || classId == 66 || classId == 67) { // remote, cell phone, microwave
                        minAreaRatio = 0.0001f; // Allow very small objects
                    }
                    // Large furniture
                    else if (classId == 56 || classId == 57 || classId == 59) { // chair, couch, dining table
                        minAreaRatio = 0.005f;  // Furniture should be reasonably sized
                        maxAreaRatio = 0.9f;
                    }
                    
                    if (width > 5 && height > 5 && areaRatio > minAreaRatio && areaRatio < maxAreaRatio && 
                        left + width <= frame.cols && top + height <= frame.rows) {
                        
                        candidates.push_back({cv::Rect(left, top, width, height), maxScore, classId});
                        
                        // Debug: Print detection info for all classes now
                        if (maxScore > 0.3f || classId != 0) { // Show non-person detections even with lower confidence
                            std::cout << "Detection: " << COCO_CLASSES[classId] << " " 
                                      << maxScore << " [" << left << "," << top << "," 
                                      << width << "," << height << "] area_ratio=" << areaRatio 
                                      << " threshold=" << classThreshold << std::endl;
                        }
                    }
                }
            }
        }

        std::cout << "Candidates before NMS: " << candidates.size() << std::endl;

        // Enhanced Non-Maximum Suppression with class-specific handling
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        std::vector<bool> removed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); i++) {
            if (removed[i]) continue;
            
            // Ensure classId is within bounds before accessing COCO_CLASSES
            if (candidates[i].classId >= 0 && candidates[i].classId < 80) {
                results.push_back({COCO_CLASSES[candidates[i].classId],
                                   candidates[i].score, candidates[i].box});
            }
            
            // Class-specific NMS thresholds
            float nmsThreshold = NMS_THRESH;
            if (candidates[i].classId == 0) { // person class
                nmsThreshold = 0.3f; // More aggressive for persons
            } else if (candidates[i].classId >= 56 && candidates[i].classId <= 60) { // furniture
                nmsThreshold = 0.5f; // Less aggressive for furniture (might have multiple similar items)
            } else if (candidates[i].classId >= 61 && candidates[i].classId <= 67) { // electronics
                nmsThreshold = 0.6f; // Even less aggressive for small electronics
            }
            
            for (size_t j = i + 1; j < candidates.size(); j++) {
                if (!removed[j]) {
                    float iouValue = iou(candidates[i].box, candidates[j].box);
                    
                    // Same class suppression with class-specific threshold
                    if (candidates[i].classId == candidates[j].classId && iouValue > nmsThreshold) {
                        removed[j] = true;
                    }
                    // Cross-class suppression only for very high overlap
                    else if (iouValue > 0.8f) {
                        removed[j] = true;
                    }
                }
            }
        }
        
        std::cout << "Final detections: " << results.size() << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
    }

    return results;
}

void VisionModule::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    std::cout << "Drawing " << detections.size() << " detections" << std::endl;
    
    for (const auto& det : detections) {
        // Class-specific colors for better visualization
        cv::Scalar color = cv::Scalar(0, 255, 0); // Default green
        
        if (det.label == "person") color = cv::Scalar(255, 0, 0); // Blue for person
        else if (det.label == "car" || det.label == "truck" || det.label == "bus") color = cv::Scalar(0, 0, 255); // Red for vehicles
        else if (det.label == "chair" || det.label == "couch" || det.label == "dining table") color = cv::Scalar(0, 165, 255); // Orange for furniture
        else if (det.label == "tv" || det.label == "laptop" || det.label == "cell phone") color = cv::Scalar(255, 255, 0); // Cyan for electronics
        else if (det.label == "bottle" || det.label == "cup" || det.label == "bowl") color = cv::Scalar(255, 0, 255); // Magenta for containers
        else if (det.label == "book" || det.label == "clock" || det.label == "vase") color = cv::Scalar(128, 0, 128); // Purple for decorative items
        
        // Adjust line thickness based on object size
        int thickness = 2;
        if (det.box.area() > frame.rows * frame.cols * 0.1) thickness = 4; // Large objects
        else if (det.box.area() < frame.rows * frame.cols * 0.01) thickness = 1; // Small objects
        
        // Draw bounding box
        cv::rectangle(frame, det.box, color, thickness);
        
        // Prepare text with confidence
        std::string text = det.label + " " + std::to_string(static_cast<int>(det.score * 100)) + "%";
        
        // Adjust text size based on box size
        double fontScale = 0.6;
        if (det.box.area() > frame.rows * frame.cols * 0.1) fontScale = 1.0;
        else if (det.box.area() < frame.rows * frame.cols * 0.01) fontScale = 0.4;
        
        // Calculate text size for background
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, 2, &baseline);
        
        // Ensure text is within frame bounds
        cv::Point textOrg = cv::Point(det.box.x, std::max(det.box.y - 5, textSize.height + 5));
        
        // Draw text background
        cv::rectangle(frame, 
                     cv::Point(textOrg.x - 2, textOrg.y - textSize.height - baseline - 2),
                     cv::Point(textOrg.x + textSize.width + 2, textOrg.y + baseline + 2),
                     color, cv::FILLED);
        
        // Draw text in contrasting color
        cv::putText(frame, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(255, 255, 255), 2);
                    
        std::cout << "Drew box for " << det.label << " (conf: " << det.score 
                  << ") at [" << det.box.x << "," << det.box.y << "," 
                  << det.box.width << "," << det.box.height << "]" << std::endl;
    }
}