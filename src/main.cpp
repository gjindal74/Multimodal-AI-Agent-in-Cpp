#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera\n";
        return -1;
    }

    cv::Mat frame;
    std::cout << "Press ESC to quit\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Webcam Test", frame);
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC
    }
    return 0;
}

