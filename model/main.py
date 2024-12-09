#  import os
#  import cv2
#  import json
#  import torch
#  import faiss
#  import numpy as np
#  from PIL import Image
#  from torchvision import transforms
#  from facenet_pytorch import MTCNN, InceptionResnetV1
#  from typing import List, Tuple, Dict
#  import csv
#  from datetime import datetime
#  import pytz
#  from collections import deque
#  from pathlib import Path
#  import time


#  def gstreamer_pipeline(
#      sensor_id=0,
#      capture_width=1920,
#      capture_height=1080,
#      display_width=960,
#      display_height=540,
#      framerate=30,
#      flip_method=0,
#  ):
#      return (
#          "nvarguscamerasrc sensor-id=%d ! "
#          "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#          "nvvidconv flip-method=%d ! "
#          "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#          "videoconvert ! "
#          "video/x-raw, format=(string)BGR ! appsink"
#          % (
#              sensor_id,
#              capture_width,
#              capture_height,
#              framerate,
#              flip_method,
#              display_width,
#              display_height,
#          )
#      )


#  class FaceRecognitionPipeline:
#      def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
#          self.device = device
#          self.mtcnn = MTCNN(
#              keep_all=True,
#              device=self.device,
#              min_face_size=120,
#              thresholds=[0.6, 0.7, 0.7],
#          )
#          self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
#          self.dimension = 512
#          self.index = faiss.IndexFlatL2(self.dimension)
#          self.transform = transforms.Compose(
#              [
#                  transforms.Resize((160, 160)),
#                  transforms.ToTensor(),
#                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#              ]
#          )
#          self.labels = []
#          self.label_ranges = {}
#          self.usernames = {}  # Store usercode -> username mapping
#          self.csv_file = "logs/recognized_users.csv"
#          self.initialize_csv()
#          self.recognized_users = set()
#          self.recognition_history = {}
#          self.recognition_threshold = 15  # Assuming 30 frames in 3 seconds (10fps)
#          self.history_length = 35  # Slightly larger than threshold to handle variations
#          self.confidence_threshold = 0.45
#          self.face_detection_times = {}  # Track how long each face has been detected
#          self.required_detection_time = 3  # 3 seconds requirement
#          self.last_log_time = {}  # Track when each user was last logged

#      def initialize_csv(self):
#          if not os.path.exists(self.csv_file):
#              with open(self.csv_file, "w", newline="") as file:
#                  writer = csv.writer(file)
#                  writer.writerow(["ID", "Username", "Timestamp (GMT+7)"])

#  def log_recognized_user(self, usercode):
#      if usercode not in self.recognized_users:
#          # Extract user details from the stored mapping
#          username = self.usernames.get(usercode, "Unknown")
#          user_dir = next((name for name in os.listdir("embedding/") if usercode in name), None)

#          if user_dir:
#              # Parse the directory name to extract all fields
#              parts = user_dir.split("_")
#              if len(parts) == 5:
#                  user_id, email, name, club, status = parts
#              else:
#                  print(f"Invalid directory format for {user_dir}")
#                  return

#              # Get the current timestamp
#              tz = pytz.timezone("Asia/Bangkok")
#              timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

#              # Append to CSV
#              with open(self.csv_file, "a", newline="") as file:
#                  writer = csv.writer(file)
#                  writer.writerow([user_id, email, name, club, status, timestamp])

#              # Add to the recognized set
#              self.recognized_users.add(usercode)

#      def load_embedding_from_json(self, json_path: str) -> np.ndarray:
#          """Load a single embedding from a JSON file"""
#          try:
#              with open(json_path, "r") as f:
#                  data = json.load(f)
#                  # Assuming the embedding is stored directly in the JSON
#                  embedding = np.array(data).astype("float32")
#                  if embedding.shape != (512,):  # Verify embedding dimension
#                      print(f"Warning: Invalid embedding dimension in {json_path}")
#                      return None
#                  return embedding
#          except Exception as e:
#              print(f"Error loading embedding from {json_path}: {e}")
#              return None

#      def add_person_from_directory(self, directory_name: str, directory_path: str):
#          """
#          Add person from a directory containing individual JSON embedding files
#          Args:
#              directory_name (str): Directory name in format "[usercode] Username"
#              directory_path (str): Path to the directory containing JSON files
#          """
#          try:
#              # Extract usercode and username from directory name
#              # Format: [usercode] Username
#              usercode = directory_name[1:].split("]")[
#                  0
#              ]  # Remove [ and get everything before ]
#              username = directory_name.split("]")[
#                  1
#              ].strip()  # Get everything after ] and strip spaces

#              # Collect all embeddings from JSON files
#              embeddings = []
#              json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]

#              for json_file in json_files:
#                  json_path = os.path.join(directory_path, json_file)
#                  embedding = self.load_embedding_from_json(json_path)
#                  if embedding is not None:
#                      embeddings.append(embedding)

#              if embeddings:
#                  start_index = self.index.ntotal
#                  embeddings_array = np.stack(embeddings)
#                  self.index.add(embeddings_array)
#                  end_index = self.index.ntotal

#                  self.label_ranges[usercode] = (start_index, end_index)
#                  self.labels.append(usercode)
#                  self.usernames[usercode] = username

#                  print(
#                      f"Added {username} ({usercode}) with {len(embeddings)} embeddings"
#                  )
#              else:
#                  print(f"No valid embeddings found for {username} ({usercode})")

#          except Exception as e:
#              print(f"Error processing directory {directory_name}: {str(e)}")

#      def add_person_from_directory_v2(self, directory_name: str, directory_path: str):
#          """
#          Add person from a directory containing individual JSON embedding files.
#          New directory format: [ID]_[Email]_[Name]_[Club]_[Status]/
#          """
#          try:
#              # Parse metadata from directory name
#              parts = directory_name.split("_")
#              if len(parts) != 5:
#                  print(f"Invalid directory format: {directory_name}")
#                  return

#              user_id, email, name, club, status = parts
#              username = f"{name} ({email})"

#              # Collect all embeddings from JSON files
#              embeddings = []
#              json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]

#              for json_file in json_files:
#                  json_path = os.path.join(directory_path, json_file)
#                  embedding = self.load_embedding_from_json(json_path)
#                  if embedding is not None:
#                      embeddings.append(embedding)

#              if embeddings:
#                  start_index = self.index.ntotal
#                  embeddings_array = np.stack(embeddings)
#                  self.index.add(embeddings_array)
#                  end_index = self.index.ntotal

#                  # Update label metadata
#                  self.label_ranges[user_id] = (start_index, end_index)
#                  self.labels.append(user_id)
#                  self.usernames[user_id] = username

#                  print(f"Added {username} (ID: {user_id}) with {len(embeddings)} embeddings")
#              else:
#                  print(f"No valid embeddings found for {username} (ID: {user_id})")

#          except Exception as e:
#              print(f"Error processing directory {directory_name}: {str(e)}")




#      def real_time_recognition(self):
#          """Start real-time face recognition using webcam"""
#          cap = cv2.VideoCapture(0)
#          if not cap.isOpened():
#              print("Error: Could not open webcam.")
#              return

#          while True:
#              ret, frame = cap.read()
#              if not ret:
#                  print("Error: Couldn't fetch the frame.")
#                  break

#              results = self.recognize_face(frame)

#              for box, label, distance, progress in results:
#                  if distance > self.confidence_threshold:
#                      label = "Unknown"

#                  color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

#                  # Draw bounding box
#                  cv2.rectangle(
#                      frame,
#                      (int(box[0]), int(box[1])),
#                      (int(box[2]), int(box[3])),
#                      color,
#                      2,
#                  )

#                  # Draw detection progress bar if face is being recognized
#                  if label != "Unknown" and progress > 0:
#                      bar_width = int(box[2] - box[0])
#                      bar_height = 5
#                      filled_width = int((progress / 100) * bar_width)

#                      # Draw background bar
#                      cv2.rectangle(
#                          frame,
#                          (int(box[0]), int(box[1] - 20)),
#                          (int(box[0] + bar_width), int(box[1] - 15)),
#                          (100, 100, 100),
#                          -1,
#                      )

#                      # Draw filled progress
#                      cv2.rectangle(
#                          frame,
#                          (int(box[0]), int(box[1] - 20)),
#                          (int(box[0] + filled_width), int(box[1] - 15)),
#                          (0, 255, 0),
#                          -1,
#                      )

#                  # Add label and confidence score
#                  cv2.putText(
#                      frame,
#                      f"{label}: {distance:.2f}",
#                      (int(box[0]), int(box[1] - 25)),
#                      cv2.FONT_HERSHEY_SIMPLEX,
#                      0.8,
#                      color,
#                      2,
#                  )

#                  # Handle recognition logging
#                  if label != "Unknown":
#                      usercode = next(
#                          (
#                              code
#                              for code, name in self.usernames.items()
#                              if name == label
#                          ),
#                          label,
#                      )
#                      if self.update_recognition_history(usercode, distance):
#                          self.log_recognized_user(usercode)

#              cv2.imshow("Real-Time Face Recognition", frame)

#              if cv2.waitKey(1) & 0xFF == ord("q"):
#                  break

#          cap.release()
#          cv2.destroyAllWindows()


#  # Main script
#  if __name__ == "__main__":
#      pipeline = FaceRecognitionPipeline()

#      home = Path.home()

#      # Add known faces from embedding directories
#      embedding_root = f"{home}/face_recognition/embedding/"

#      for user_folder in os.listdir(embedding_root):
#          folder_path = os.path.join(embedding_root, user_folder)
#          if os.path.isdir(folder_path):
#              print(f"Processing {user_folder}...")
#              pipeline.add_person_from_directory(user_folder, folder_path)

#      # Start real-time face recognition
#      pipeline.real_time_recognition()

import os
import cv2
import json
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import List, Tuple, Dict
import csv
from datetime import datetime
import pytz
from collections import deque
from pathlib import Path
import time
from PIL import ImageFont, ImageDraw, Image

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

#  def gstreamer_pipeline(
#      sensor_id=0,
#      capture_width=1920,
#      capture_height=1080,
#      display_width=960,
#      display_height=540,
#      framerate=30,
#      flip_method=0,
#  ):
#      return (
#          "nvarguscamerasrc sensor-id=%d ! "
#          "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#          "nvvidconv flip-method=%d ! "
#          "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#          "videoconvert ! "
#          "video/x-raw, format=(string)BGR ! appsink"
#          % (
#              sensor_id,
#              capture_width,
#              capture_height,
#              framerate,
#              flip_method,
#              display_width,
#              display_height,
#          )
#      )


class FaceRecognitionPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=120,
            thresholds=[0.7, 0.7, 0.7],
        )
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.dimension = 512
        self.index = faiss.IndexFlatL2(self.dimension)
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.labels = []
        self.label_ranges = {}
        self.user_metadata = {}  # Store user details (ID, Email, Name, Club, Status)
        self.csv_file = "logs/recognized_users.csv"
        self.initialize_csv()
        self.recognized_users = set()
        self.recognition_history = {}
        self.recognition_threshold = 15  # Assuming 30 frames in 3 seconds (10fps)
        self.history_length = 35  # Slightly larger than threshold to handle variations
        self.confidence_threshold = 0.45
        self.face_detection_times = {}  # Track how long each face has been detected
        self.required_detection_time = 3  # 3 seconds requirement
        self.last_log_time = {}  # Track when each user was last logged

    def initialize_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["ID", "Email", "Name", "Club", "Status", "Timestamp (GMT+7)"]
                )

    def log_recognized_user(self, usercode):
        if usercode not in self.recognized_users:
            tz = pytz.timezone("Asia/Bangkok")
            timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            user_details = self.user_metadata.get(usercode, {})
            with open(self.csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        user_details.get("ID", ""),
                        user_details.get("Email", ""),
                        user_details.get("Name", ""),
                        user_details.get("Club", ""),
                        user_details.get("Status", ""),
                        timestamp,
                    ]
                )
            self.recognized_users.add(usercode)

    def load_embedding_from_json(self, json_path: str) -> np.ndarray:
        """Load a single embedding from a JSON file"""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                embedding = np.array(data).astype("float32")
                if embedding.shape != (512,):
                    print(f"Warning: Invalid embedding dimension in {json_path}")
                    return None
                return embedding
        except Exception as e:
            print(f"Error loading embedding from {json_path}: {e}")
            return None

    def add_person_from_directory_v2(self, directory_name: str, directory_path: str):
        """
        Add person from a directory containing embeddings and user metadata
        Format: [ID]_[Email]_[Name]_[Club]_[Status]
        """
        try:
            # Remove the outer square brackets and split
            cleaned_name = directory_name.replace('[', '').replace(']', '')
            parts = cleaned_name.split('_')
            
            if len(parts) != 5:
                print(f"Error: Invalid directory format for {directory_name}")
                return

            user_metadata = {
                "ID": parts[0],
                "Email": parts[1],
                "Name": parts[2],
                "Club": parts[3],
                "Status": parts[4]
            }

            usercode = user_metadata["ID"]
            self.user_metadata[usercode] = user_metadata

            # Collect all embeddings
            embeddings = []
            json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]

            for json_file in json_files:
                json_path = os.path.join(directory_path, json_file)
                embedding = self.load_embedding_from_json(json_path)
                if embedding is not None:
                    embeddings.append(embedding)

            if embeddings:
                start_index = self.index.ntotal
                embeddings_array = np.stack(embeddings)
                self.index.add(embeddings_array)
                end_index = self.index.ntotal

                self.label_ranges[usercode] = (start_index, end_index)
                self.labels.append(usercode)

                print(f"Added user {user_metadata['Name']} with {len(embeddings)} embeddings")
            else:
                print(f"No valid embeddings found for {directory_name}")

        except Exception as e:
            print(f"Error processing directory {directory_name}: {str(e)}")
    # Existing methods (detect_faces, recognize_face, real_time_recognition) remain unchanged

    def real_time_recognition(self):
        """Start real-time face recognition using webcam"""
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        #  # Create a window for full screen
        #  cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
        #  cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, 1)
        font_path = "assets/dejavu-sans/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 20)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't fetch the frame.")
                break

            # Get recognition results
            results = self.recognize_face(frame)
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()

            # First draw all OpenCV elements (boxes and progress bars)
            for box, usercode, distance, progress in results:
                if distance > self.confidence_threshold:
                    display_label = "Unknown"
                    color = (0, 0, 255)  # BGR format for OpenCV
                else:
                    display_label = self.user_metadata.get(usercode, {}).get("Name", "Unknown")
                    color = (0, 255, 0)  # BGR format for OpenCV

                # Draw bounding box
                cv2.rectangle(
                    display_frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2
                )

                # Draw progress bar
                bar_width = int(box[2] - box[0])
                bar_height = 5
                filled_width = int((progress / 100) * bar_width)

                # Background bar
                cv2.rectangle(
                    display_frame,
                    (int(box[0]), int(box[1] - 20)),
                    (int(box[0] + bar_width), int(box[1] - 15)),
                    (100, 100, 100),
                    -1
                )

                # Filled progress
                if filled_width > 0:
                    cv2.rectangle(
                        display_frame,
                        (int(box[0]), int(box[1] - 20)),
                        (int(box[0] + filled_width), int(box[1] - 15)),
                        (0, 255, 0),
                        -1
                    )

                # Handle recognition logging
                if display_label != "Unknown":
                    if self.update_recognition_history(usercode, distance):
                        self.log_recognized_user(usercode)

            # Convert to PIL for text rendering
            pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Draw text on PIL image
            for box, usercode, distance, progress in results:
                if distance > self.confidence_threshold:
                    display_label = "Unknown"
                    color = (255, 0, 0)  # RGB format for PIL
                else:
                    display_label = self.user_metadata.get(usercode, {}).get("Name", "Unknown")
                    color = (0, 255, 0)  # RGB format for PIL

                text = f"{display_label}: {distance:.2f}"
                draw.text(
                    (int(box[0]), int(box[1] - 40)),
                    text,
                    font=font,
                    fill=color
                )

            # Convert back to OpenCV format for display
            final_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Real-Time Face Recognition", final_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes, _ = self.mtcnn.detect(image)
        return boxes

    def recognize_face(
        self, image: np.ndarray, threshold: float = 0.7
    ) -> List[Tuple[Tuple[int, int, int, int], str, float, float]]:
        boxes = self.detect_faces(image)
        results = []
        current_time = time.time()

        # Clear detection times for faces that are no longer visible
        detected_labels = set()

        if boxes is not None:
            embeddings = []
            for box in boxes:
                face = Image.fromarray(image).crop(box)
                embedding = self.get_embedding(face)
                embeddings.append(embedding[0])

            if embeddings:
                distances, indices = self.index.search(np.array(embeddings), 1)
                for i, (distance, index) in enumerate(zip(distances, indices)):
                    recognized_label = "Unknown"
                    if distance[0] < threshold:
                        for label, (start, end) in self.label_ranges.items():
                            if start <= index[0] < end:
                                recognized_label = label
                                detected_labels.add(recognized_label)
                                break

                    # Calculate detection progress (0 to 100%)
                    detection_progress = 0
                    if recognized_label != "Unknown":
                        detection_start = self.face_detection_times.get(
                            recognized_label, current_time
                        )
                        detection_duration = current_time - detection_start
                        detection_progress = min(
                            100,
                            (detection_duration / self.required_detection_time) * 100,
                        )

                    results.append(
                        (boxes[i], recognized_label, distance[0], detection_progress)
                    )

        # Clear detection times for faces no longer in frame
        for label in list(self.face_detection_times.keys()):
            if label not in detected_labels:
                self.face_detection_times.pop(label, None)
                self.recognition_history.pop(label, None)

        return results

    def get_embedding(self, face_image: Image.Image) -> np.ndarray:
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
        return embedding.cpu().numpy()

    def update_recognition_history(self, label: str, confidence: float) -> bool:
        current_time = time.time()

        # Initialize history if not exists
        if label not in self.recognition_history:
            self.recognition_history[label] = deque(maxlen=self.history_length)
            self.face_detection_times[label] = current_time
            return False

        # Add new confidence score
        self.recognition_history[label].append(confidence)

        # Check if we have enough consecutive detections
        if len(self.recognition_history[label]) >= self.recognition_threshold:
            recent_recognitions = list(self.recognition_history[label])[
                -self.recognition_threshold :
            ]

            # Check if all recent detections are confident enough
            if all(conf < self.confidence_threshold for conf in recent_recognitions):
                # Check if face has been detected for required time
                detection_duration = current_time - self.face_detection_times[label]

                # Check if enough time has passed since last logging (prevent duplicate logs)
                last_log = self.last_log_time.get(label, 0)
                if (
                    detection_duration >= self.required_detection_time
                    and (current_time - last_log) > 5
                ):  # 5 second cooldown
                    self.last_log_time[label] = current_time
                    return True

        return False
# Main script
if __name__ == "__main__":
    pipeline = FaceRecognitionPipeline()

    home = Path.home()
    embedding_root = f"{home}/face_recognition/embedding/"

    for user_folder in os.listdir(embedding_root):
        folder_path = os.path.join(embedding_root, user_folder)
        if os.path.isdir(folder_path):
            print(f"Processing {user_folder}...")
            pipeline.add_person_from_directory_v2(user_folder, folder_path)

    pipeline.real_time_recognition()
