import cv2
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
# Open the camera
# cap = cv2.VideoCapture(0)

cap  = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Initialize variables for FPS calculation
total_capture_time = 0  # Cumulative capture time in seconds
frame_count = 0         # Number of frames captured in the current period
fps = 0                 # FPS value

while True:
    # Measure time before capturing frame
    capture_start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Measure time after capturing frame
    capture_end_time = time.time()

    # Calculate capture time for this frame
    capture_time = capture_end_time - capture_start_time

    # Accumulate capture time and increment frame count
    total_capture_time += capture_time
    
    frame_count += 1.0

    # Check if total capture time is >= 1 second
    if total_capture_time >= 1.0:
        # Calculate FPS
        fps = frame_count / total_capture_time
        # print(f"Capture FPS: {fps:.2f}")
        # Reset for the next period
        total_capture_time = 0
        frame_count = 0

    # Display FPS on the frame
    # new_time = time.time()
    fps_text = f"Capture FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # new_end_time = time.time()

    # print(f"Time to display FPS: {new_end_time - new_time:.5f} seconds")
    # Show the frame
    cv2.imshow('Camera', frame)

    # Log the capture time for this frame
    # print(f"Frame capture time: {capture_time:.5f} seconds")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
