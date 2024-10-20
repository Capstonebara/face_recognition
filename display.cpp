#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    // Open the default camera (index 0)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Couldn't open video stream" << endl;
        return -1;
    }

    // Create a window to display the video
    namedWindow("Video Feed", WINDOW_NORMAL);

    // Variables to calculate FPS
    auto last_time = chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0.0;

    // Get the video frame width and height
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Set the desired size for the larger frame (e.g., double the size)
    int larger_width = frame_width * 2;
    int larger_height = frame_height * 2;

    // Check if CUDA is available
    if (!cuda::getCudaEnabledDeviceCount()) {
        cout << "CUDA not available. Falling back to CPU" << endl;
    } else {
        cout << "CUDA enabled. Using GPU for processing" << endl;
    }

    while (true) {
        Mat frame;
        cap >> frame;  // Capture a frame

        if (frame.empty()) {
            cout << "Error: Couldn't capture the frame" << endl;
            break;
        }

        // Upload the frame to GPU
        cuda::GpuMat gpu_frame, gpu_resized_frame;
        gpu_frame.upload(frame);

        // Resize the frame on the GPU
        cuda::resize(gpu_frame, gpu_resized_frame, Size(larger_width, larger_height));

        // Download the resized frame back to CPU for displaying
        Mat larger_frame;
        gpu_resized_frame.download(larger_frame);

        // Calculate FPS
        frame_count++;
        auto current_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(current_time - last_time).count();

        if (duration >= 1000) {  // If 1 second has passed
            fps = frame_count * 1000.0 / duration;
            last_time = current_time;
            frame_count = 0;
        }

        // Display FPS on the frame
        string fps_text = "FPS: " + to_string(fps);
        putText(larger_frame, fps_text, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // Display the resized frame in the "Video Feed" window
        imshow("Video Feed", larger_frame);

        // Wait for 1ms and check if the 'q' key is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and destroy the window
    cap.release();
    destroyAllWindows();

    return 0;
}
