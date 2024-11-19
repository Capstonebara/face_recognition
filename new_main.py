import os
import cv2
import json
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from typing import List, Tuple, Dict
import csv
from datetime import datetime
import pytz
from collections import deque

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class FaceRecognitionPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Comment out MTCNN initialization
        # self.mtcnn = MTCNN(
        #     keep_all=True,
        #     device=self.device,
        #     min_face_size=120,
        #     thresholds=[0.6, 0.7, 0.7],
        # )
        
        # Initialize Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
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
        self.usernames = {}
        self.csv_file = "recognized_users.csv"
        self.initialize_csv()
        self.recognized_users = set()
        self.recognition_history = {}
        self.recognition_threshold = 5
        self.history_length = 10
        self.confidence_threshold = 0.6

    def initialize_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Username", "Timestamp (GMT+7)"])

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # boxes, _ = self.mtcnn.detect(image)
        # return boxes
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Convert to the same format as MTCNN output (x1, y1, x2, y2)
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
        
        return np.array(boxes) if len(boxes) > 0 else None

    # Rest of the methods remain the same
    def log_recognized_user(self, usercode):
        if usercode not in self.recognized_users:
            tz = pytz.timezone("Asia/Bangkok")
            timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            username = self.usernames.get(usercode, usercode)
            with open(self.csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([usercode, username, timestamp])
            self.recognized_users.add(usercode)

    def load_embedding_from_json(self, json_path: str) -> np.ndarray:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                embedding = np.array(data).astype('float32')
                if embedding.shape != (512,):
                    print(f"Warning: Invalid embedding dimension in {json_path}")
                    return None
                return embedding
        except Exception as e:
            print(f"Error loading embedding from {json_path}: {e}")
            return None

    def add_person_from_directory(self, directory_name: str, directory_path: str):
        try:
            usercode = directory_name[1:].split("]")[0]
            username = directory_name.split("]")[1].strip()
            
            embeddings = []
            json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
            
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
                self.usernames[usercode] = username
                
                print(f"Added {username} ({usercode}) with {len(embeddings)} embeddings")
            else:
                print(f"No valid embeddings found for {username} ({usercode})")
                
        except Exception as e:
            print(f"Error processing directory {directory_name}: {str(e)}")

    def recognize_face(
        self, image: np.ndarray, threshold: float = 0.7
    ) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        boxes = self.detect_faces(image)
        results = []
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
                                break
                    results.append((boxes[i], recognized_label, distance[0]))
        return results
    
    def update_recognition_history(self, label: str, confidence: float) -> bool:
        if label not in self.recognition_history:
            self.recognition_history[label] = deque(maxlen=self.history_length)
        self.recognition_history[label].append(confidence)
        if len(self.recognition_history[label]) >= self.recognition_threshold:
            recent_recognitions = list(self.recognition_history[label])[-self.recognition_threshold:]
            if all(conf < self.confidence_threshold for conf in recent_recognitions):
                return True
        return False

    def get_embedding(self, face_image: Image.Image) -> np.ndarray:
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
        return embedding.cpu().numpy()

    def real_time_recognition(self):
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        window_title = "Real-Time Face Recognition"
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't fetch the frame.")
                break

            results = self.recognize_face(frame)
            
            for box, label, distance in results:
                if distance > self.confidence_threshold:
                    label = "Unknown"
                
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
                
                cv2.putText(
                    frame,
                    f"{label}: {distance:.2f}",
                    (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                
                if label != "Unknown":
                    usercode = next((code for code, name in self.usernames.items() if name == label), label)
                    if self.update_recognition_history(usercode, distance):
                        self.log_recognized_user(usercode)
            
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(window_title, frame)
            else:
                break
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    pipeline = FaceRecognitionPipeline()

    # Add known faces from embedding directories
    embedding_root = "/home/jetson/face_recognition/embedding/"
    
    for user_folder in os.listdir(embedding_root):
        folder_path = os.path.join(embedding_root, user_folder)
        if os.path.isdir(folder_path):
            print(f"Processing {user_folder}...")
            pipeline.add_person_from_directory(user_folder, folder_path)

    # Start real-time face recognition
    pipeline.real_time_recognition()
