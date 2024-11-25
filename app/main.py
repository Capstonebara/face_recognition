import sys
import os
import cv2
import pandas as pd
import time
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path
import csv

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from model.test_main import FaceRecognitionPipeline
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QTableView,
    QSizePolicy,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QAbstractTableModel, QSize
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FileWatcher(QThread):
    file_changed = pyqtSignal(pd.DataFrame)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.running = True
        self.last_modified_time = 0
        self.headers = ["ID", "Email", "Name", "Club", "Status", "Timestamp (GMT+7)"]
        self.headers_added = False  # Flag to track if we've logged the headers message
        self.initialize_csv()

    def initialize_csv(self):
        """Initialize or reinitialize the CSV file with headers"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # Check if file exists
            file_exists = os.path.exists(self.file_path)
            
            if not file_exists:
                # File doesn't exist - create new with headers
                if not self.headers_added:
                    print("Creating new CSV file with headers...")
                    self.headers_added = True
                with open(self.file_path, "w", newline="", encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.headers)
                return

            # File exists - read content
            try:
                df = pd.read_csv(self.file_path, encoding='utf-8')
                
                # Check if file is empty (no data rows)
                if df.empty:
                    if not self.headers_added:
                        print("File is empty. Adding headers...")
                        self.headers_added = True
                    with open(self.file_path, "w", newline="", encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(self.headers)
                    return
                
                # Check if headers match expected headers
                if list(df.columns) != self.headers:
                    print("Headers don't match expected format. Fixing...")
                    # Create backup of existing file
                    backup_path = f"{self.file_path}.backup"
                    os.rename(self.file_path, backup_path)
                    print(f"Created backup at {backup_path}")
                    
                    # Create new file with correct headers and existing data
                    with open(self.file_path, "w", newline="", encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(self.headers)
                        
                        # If there was any data in the original file, write it back
                        if len(df) > 0:
                            df.to_csv(self.file_path, mode='a', header=False, index=False)
                    
            except pd.errors.EmptyDataError:
                # File exists but is completely empty
                if not self.headers_added:
                    print("File exists but is empty. Adding headers...")
                    self.headers_added = True
                with open(self.file_path, "w", newline="", encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.headers)

        except Exception as e:
            print(f"Error initializing CSV: {e}")

    def run(self):
        while self.running:
            try:
                # Always ensure file exists with headers before processing
                if not os.path.exists(self.file_path):
                    self.initialize_csv()
                    continue

                # Read file to check if it's properly formatted
                try:
                    df = pd.read_csv(self.file_path, encoding='utf-8')
                    
                    # If file is empty or headers don't match, reinitialize
                    if df.empty or list(df.columns) != self.headers:
                        self.initialize_csv()
                        continue
                        
                    current_mtime = os.path.getmtime(self.file_path)
                    if current_mtime > self.last_modified_time:
                        self.file_changed.emit(df)
                        self.last_modified_time = current_mtime
                        
                except pd.errors.EmptyDataError:
                    self.initialize_csv()
                    continue

            except Exception as e:
                print(f"Error reading file: {e}")
                self.initialize_csv()
            
            self.msleep(100)  # Short sleep to prevent high CPU usage

    def stop(self):
        self.running = False


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    face_detected_signal = pyqtSignal(str)

    def __init__(self, face_recognition_model):
        super().__init__()
        self.running = True
        self.model = face_recognition_model
        self.cap = None
        # Initialize font for text rendering
        try:
            self.font = ImageFont.truetype("assets/dejavu-sans/DejaVuSans.ttf", 20)
        except Exception as e:
            print(f"Error loading font: {e}")
            self.font = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.model:
                    # Get recognition results
                    results = self.model.recognize_face(frame)
                    
                    # Create a copy of the frame for drawing
                    display_frame = frame.copy()

                    # First draw all OpenCV elements (boxes and progress bars)
                    for box, label, distance, progress in results:
                        if distance > self.model.confidence_threshold:
                            display_label = "Unknown"
                            color = (0, 0, 255)  # BGR format for OpenCV
                        else:
                            display_label = self.model.user_metadata.get(label, {}).get("Name", "Unknown")
                            color = (0, 255, 0)  # BGR format for OpenCV

                        # Draw bounding box
                        cv2.rectangle(
                            display_frame,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color,
                            2
                        )

                        # Draw detection progress bar if face is being recognized
                        if label != "Unknown" and progress > 0:
                            bar_width = int(box[2] - box[0])
                            bar_height = 5
                            filled_width = int((progress / 100) * bar_width)

                            # Draw background bar
                            cv2.rectangle(
                                display_frame,
                                (int(box[0]), int(box[1] - 20)),
                                (int(box[0] + bar_width), int(box[1] - 15)),
                                (100, 100, 100),
                                -1
                            )

                            # Draw filled progress
                            cv2.rectangle(
                                display_frame,
                                (int(box[0]), int(box[1] - 20)),
                                (int(box[0] + filled_width), int(box[1] - 15)),
                                (0, 255, 0),
                                -1
                            )

                        # Handle recognition logging
                        if label != "Unknown":
                            if self.model.update_recognition_history(label, distance):
                                self.model.log_recognized_user(label)
                                self.face_detected_signal.emit(label)

                    # Convert to PIL for text rendering
                    if self.font:
                        pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)

                        # Draw text on PIL image
                        for box, label, distance, progress in results:
                            if distance > self.model.confidence_threshold:
                                display_label = "Unknown"
                                color = (255, 0, 0)  # RGB format for PIL
                            else:
                                display_label = self.model.user_metadata.get(label, {}).get("Name", "Unknown")
                                color = (0, 255, 0)  # RGB format for PIL

                            text = f"{display_label}: {distance:.2f}"
                            draw.text(
                                (int(box[0]), int(box[1] - 40)),
                                text,
                                font=self.font,
                                fill=color
                            )

                        # Convert back to OpenCV format
                        display_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    self.change_pixmap_signal.emit(display_frame)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class CameraWindow(QMainWindow):
    def __init__(self, face_recognition_model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera View")

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create image label with size policy
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(QSize(640, 480))  # Set minimum size
        layout.addWidget(self.image_label)

        # Set window properties
        self.setMinimumSize(QSize(800, 600))  # Set minimum window size

        # Start video thread
        self.thread = VideoThread(face_recognition_model)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, cv_img):
        """
        This method updates the image to be displayed in the label.
        """
        qt_img = self.convert_cv_qt(cv_img)
        self.display_resized_image(qt_img)

    def convert_cv_qt(self, cv_img):
        """
        Convert OpenCV image (BGR) to QImage (RGB) for display.
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def display_resized_image(self, qt_img):
        """
        Resize the pixmap to fit the image label and display it.
        """
        scaled_pixmap = qt_img.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """
        Handle window resize. This will ensure the image updates
        when the window size changes.
        """
        super().resizeEvent(event)
        # We don't need to resize the pixmap again here; it's handled in update_image

    def closeEvent(self, event):
        """
        Stop the video thread when the window is closed.
        """
        self.thread.stop()
        event.accept()


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data  # Remove the max_rows

    def rowCount(self, parent=None):
        return len(self._data)
        
    def columnCount(self, parent=None):
        return len(self._data.columns)
        
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        return None
        
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.face_recognition_model = FaceRecognitionPipeline()

        home = Path.home()
        embedding_root = f"{home}/face_recognition/embedding/"

        for user_folder in os.listdir(embedding_root):
            folder_path = os.path.join(embedding_root, user_folder)
            if os.path.isdir(folder_path):
                print(f"Processing {user_folder}...")
                self.face_recognition_model.add_person_from_directory_v2(
                    user_folder, folder_path
                )

        self.camera_window = None
        self.file_watcher = None
        self.table_view = None

        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        button_layout = QHBoxLayout()
        self.camera_btn = QPushButton("Open Camera")
        self.camera_btn.clicked.connect(self.open_camera)
        button_layout.addWidget(self.camera_btn)
        layout.addLayout(button_layout)

        self.table_view = QTableView()
        layout.addWidget(self.table_view)

        # Start watching the recognized_users.csv file
        csv_path = os.path.join(project_root, "logs", "recognized_users.csv")
        self.start_file_watcher(csv_path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.table_view:
            self.resize_table_content()

    def resize_table_content(self):
        """
        Resize the table columns and rows to fit the available widget space dynamically.
        """
        self.table_view.resizeColumnsToContents()
        self.table_view.resizeRowsToContents()

        # Optionally, you can use the stretch modes for columns to fill available space:
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

    def start_file_watcher(self, file_path):
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.wait()

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.file_watcher = FileWatcher(file_path)
        self.file_watcher.file_changed.connect(self.update_table)
        self.file_watcher.start()

        try:
            if not os.path.exists(file_path):
                # Create empty CSV with headers
                df = pd.DataFrame(columns=["ID", "Email", "Name", "Club", "Status"])
                df.to_csv(file_path, index=False)
            elif os.path.getsize(file_path) > 0:
                if Path(file_path).suffix == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            else:
                df = pd.DataFrame(columns=["ID", "Email", "Name", "Club", "Status"])

            self.update_table(df)
        except Exception as e:
            print(f"Error loading initial data: {e}")

    def update_table(self, df):
        # Create model with the dataset
        model = PandasModel(df)
        self.table_view.setModel(model)

        # Adjust table size on content update
        self.resize_table_content()

        # Scroll to the bottom to show the newest entry
        self.table_view.scrollToBottom()

    def open_camera(self):
        if self.camera_window is not None:
            self.camera_window.close()
            self.camera_window.deleteLater()
            self.camera_window = None

        self.camera_window = CameraWindow(self.face_recognition_model, self)
        self.camera_window.show()

    def closeEvent(self, event):
        if self.camera_window:
            self.camera_window.close()
            self.camera_window.deleteLater()
            self.camera_window = None
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
