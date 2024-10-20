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


class FaceRecognitionPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=120,
            thresholds=[0.6, 0.7, 0.7],
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
        self.csv_file = "recognized_users.csv"
        self.initialize_csv()
        self.recognized_users = set()  # Initialize this attribute
        self.recognition_history: Dict[str, deque] = {}
        self.recognition_threshold = 5
        self.history_length = 10
        self.confidence_threshold = 0.6

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes, _ = self.mtcnn.detect(image)
        return boxes

    def get_embedding(self, face_image: Image.Image) -> np.ndarray:
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
        return embedding.cpu().numpy()

    def add_person_from_embeddings(self, name: str, json_file_path: str):
        start_index = self.index.ntotal
        embeddings, labels = load_precomputed_embeddings(json_file_path)
        if embeddings is not None:
            self.index.add(embeddings)
            end_index = self.index.ntotal
            self.label_ranges[name] = (start_index, end_index)
            self.labels.append(name)

    def add_person(self, name, image_folder):
        start_index = self.index.ntotal
        embeddings = []
        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            boxes = self.detect_faces(np.array(image))
            if boxes is not None:
                for box in boxes:
                    face = image.crop(box)
                    embedding = self.get_embedding(face)
                    embeddings.append(embedding[0])
        if embeddings:
            self.index.add(np.array(embeddings))
            end_index = self.index.ntotal
            self.label_ranges[name] = (start_index, end_index)
            self.labels.append(name)

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
            recent_recognitions = list(self.recognition_history[label])[
                -self.recognition_threshold :
            ]
            if all(conf < self.confidence_threshold for conf in recent_recognitions):
                return True
        return False

    def initialize_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Timestamp (GMT+7)"])

    def log_recognized_user(self, user_id):
        if user_id not in self.recognized_users:
            tz = pytz.timezone("Asia/Bangkok")
            timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([user_id, timestamp])
            self.recognized_users.add(user_id)

    def real_time_recognition(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
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
                    if self.update_recognition_history(label, distance):
                        self.log_recognized_user(label)
            cv2.imshow("Real-Time Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def load_precomputed_embeddings(json_file_path: str) -> Tuple[np.ndarray, List[str]]:
    embeddings = []
    labels = []
    with open(json_file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading {json_file_path}: {e}")
            return None, None
        for image_name, embedding_data in data.items():
            embeddings.append(embedding_data["vector"])
            labels.append(image_name)
    if embeddings:
        embeddings_np = np.array(embeddings).astype("float32")
        return embeddings_np, labels
    else:
        print("No embeddings found.")
        return None, None


# Main script to run the face recognition pipeline
if __name__ == "__main__":
    pipeline = FaceRecognitionPipeline()

    # Add known faces from precomputed embeddings stored in JSON files
    json_folder_path = "/home/vanellope/face_recognition_project/embedding/"

    for filename in os.listdir(json_folder_path):
        if filename.endswith("_embeddings.json"):
            name = filename.split("_embeddings")[0]
            json_file = os.path.join(json_folder_path, filename)
            print(f"Adding {name} from {json_file}...")
            pipeline.add_person_from_embeddings(name, json_file)
    #  assets_directory = "/home/vanellope/face_recognition_project/assets"
    #  people = [
    #      (folder_name, os.path.join(assets_directory, folder_name))
    #      for folder_name in os.listdir(assets_directory)
    #      if os.path.isdir(os.path.join(assets_directory, folder_name))
    #  ]

    #  # Add known faces to the database using a loop
    #  for name, folder in people:
    #      print(f"Adding {name} from {folder}...")
    #      pipeline.add_person(name, folder)
    # Start real-time face recognition
    pipeline.real_time_recognition()
