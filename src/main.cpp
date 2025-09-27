#include "vision/visionmodule.h"
#include <opencv2/opencv.hpp>

int main() {
    VisionModule vision("/Users/gaurangjindal/Desktop/multimodal-agent-cpp/models/yolov8n.onnx");

    if (!vision.init()) return -1;

    cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        auto dets = vision.detect(frame);
        vision.drawDetections(frame, dets);

        cv::imshow("Detections", frame);
        if (cv::waitKey(1) == 27) break; // ESC
    }
    return 0;
}
