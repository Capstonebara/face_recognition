import cv2
import time
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start the video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Initialize the counter for image filenames
image_counter = 0


# Function to capture the frame and save the image if a face is detected
def capture_frame():
    global image_counter

    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        return False

    # Convert the frame to grayscale (Haar Cascade needs grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # If faces are detected, save the image
    if len(faces) > 0:
        print(f"Detected {len(faces)} face(s), saving the image...")

        # Draw a rectangle around the faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        user = "hung"

        # Define the path for saving the image
        folder_path = f"/home/vanellope/face_recognition_project/assets/{user}/"

        # Create the user folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the captured image with a counter-based filename (e.g., người lợn0.jpg, người lợn1.jpg)
        filename = f"{folder_path}{user}+{image_counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

        # Increment the counter for the next image
        image_counter += 1

        return True

    return False


# Capture images every 2 seconds until 'q' is pressed
try:
    while True:
        # Capture and save a frame if a face is detected
        capture_frame()

        # Wait for 2 seconds before capturing the next frame
        time.sleep(2)

        # Check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nScript terminated by user.")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
