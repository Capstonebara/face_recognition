# import cv2

# # Read an image (change the path to your image file)
# img = cv2.imread("/home/vanellope/face_recognition_project/assets/obama/obama0.jpeg")

# # Show the image in a window
# cv2.imshow("Image", img)

# # Wait for any key to close the window
# cv2.waitKey(0)

# # Close all OpenCV windows
# cv2.destroyAllWindows()
import cv2


def test_video_stream():
    # Open the video capture (0 for default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open video stream.")
        return

    # Start video capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Display the video feed in a window
        cv2.imshow("Video Stream", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Test the video stream
if __name__ == "__main__":
    test_video_stream()
