import sys
import os
import cv2
import pandas as pd
import time
from pathlib import Path
import csv

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from model.main import FaceRecognitionPipeline
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
        self.initialize_csv()

    def initialize_csv(self):
        if not os.path.exists(self.file_path):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # Write headers to a new CSV file
            with open(self.file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Email", "Name", "Club", "Status"])

                # Add a sample row to avoid the file being completely empty
                writer.writerow(
                    [
                        "1",
                        "System",
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 7 * 3600)
                        ),  # GMT+7
                    ]
                )

        # Handle blank but existing files
        elif os.path.getsize(self.file_path) == 0:
            with open(self.file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Email", "Name", "Club", "Status"])

    def run(self):
        self.initialize_csv()  # Ensure CSV exists when thread starts
        while self.running:
            try:
                current_mtime = os.path.getmtime(self.file_path)
                if current_mtime > self.last_modified_time:
                    df = pd.read_csv(self.file_path)
                    self.file_changed.emit(df)
                    self.last_modified_time = current_mtime
            except Exception as e:
                print(f"Error reading file: {e}")
            self.msleep(100)

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

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.model:
                    results = self.model.recognize_face(frame)
                    for box, label, distance, progress in results:
                        if distance > self.model.confidence_threshold:
                            label = "Unknown"

                        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                        cv2.rectangle(
                            frame,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color,
                            2,
                        )

                        # Draw detection progress bar if face is being recognized
                        if label != "Unknown" and progress > 0:
                            bar_width = int(box[2] - box[0])
                            bar_height = 5
                            filled_width = int((progress / 100) * bar_width)

                            # Draw background bar
                            cv2.rectangle(
                                frame,
                                (int(box[0]), int(box[1] - 20)),
                                (int(box[0] + bar_width), int(box[1] - 15)),
                                (100, 100, 100),
                                -1,
                            )

                            # Draw filled progress
                            cv2.rectangle(
                                frame,
                                (int(box[0]), int(box[1] - 20)),
                                (int(box[0] + filled_width), int(box[1] - 15)),
                                (0, 255, 0),
                                -1,
                            )

                        cv2.putText(
                            frame,
                            f"{label}: {distance:.2f}",
                            (int(box[0]), int(box[1] - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2,
                        )

                        if label != "Unknown":
                            usercode = next(
                                (
                                    code
                                    for code, name in self.model.usernames.items()
                                    if name == label
                                ),
                                label,
                            )
                            if self.model.update_recognition_history(
                                usercode, distance
                            ):
                                self.model.log_recognized_user(usercode)
                                self.face_detected_signal.emit(usercode)

                self.change_pixmap_signal.emit(frame)
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
    def __init__(self, data, max_rows=4):
        super().__init__()
        self.max_rows = max_rows
        self._data = self.process_data(data)

    def process_data(self, data):
        # If the dataframe has more rows than max_rows, keep only the most recent ones
        if len(data) > self.max_rows:
            return data.iloc[-self.max_rows :]
        return data

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
                self.face_recognition_model.add_person_from_directory(
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
