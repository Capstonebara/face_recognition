#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>

int main() {
    // Open the camera
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Error: Could not open video device." << std::endl;
        return -1;
    }

    // Initialize variables for FPS calculation
    float total_capture_time = 0.0f;  // Cumulative capture time in seconds
    float frame_count = 0.0f;         // Number of frames captured in the current period
    float fps = 0.0f;                 // FPS value

    while (true) {
        // Measure time before capturing frame
        auto capture_start_time = std::chrono::steady_clock::now();

        // Capture frame-by-frame
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            std::cout << "Error: Failed to capture image." << std::endl;
            break;
        }

        // Measure time after capturing frame
        auto capture_end_time = std::chrono::steady_clock::now();

        // Calculate capture time for this frame
        std::chrono::duration<float> capture_duration = capture_end_time - capture_start_time;
        float capture_time = capture_duration.count();

        // Accumulate capture time and increment frame count
        total_capture_time += capture_time;
        frame_count += 1.0f;

        // Check if total capture time is >= 1 second
        if (total_capture_time >= 1.0f) {
            // Calculate FPS
            fps = frame_count / total_capture_time;
            // Reset for the next period
            total_capture_time = 0.0f;
            frame_count = 0.0f;
        }

        // Display FPS on the frame
        std::string fps_text = "Capture FPS: " + std::to_string(static_cast<int>(fps * 100.0f) / 100.0f);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                    1.0f, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // Show the frame
        cv::imshow("Camera", frame);

        // Press 'q' to quit - fixed the precedence warning
        if ((cv::waitKey(1) & 0xFF) == 'q') {
            break;
        }
    }

    // Release the camera and close all OpenCV windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}