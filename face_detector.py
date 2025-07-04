import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

# Worker class to run face detection in a separate thread
class FaceDetectionWorker(QThread):
    # Signal to send the processed frame back to the UI
    processed_frame = pyqtSignal(QImage)
    # Signal to display error messages
    error_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.face_cascade = None
        self.cap = None

    def run(self):
        # Load the Haar Cascade classifier for face detection
        # Make sure the 'haarcascade_frontalface_default.xml' file is in the same directory
        # or provide its full path.
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                self.error_message.emit("Failed to load haarcascade_frontalface_default.xml. Ensure the file is in the correct location.")
                return
        except Exception as e:
            self.error_message.emit(f"Error loading cascade classifier: {e}")
            return

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0) # 0 is the default ID for the webcam
        if not self.cap.isOpened():
            self.error_message.emit("Failed to open webcam. Ensure the webcam is connected and not in use by another application.")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.error_message.emit("Failed to read frame from webcam.")
                break

            # Convert the frame to grayscale (required for face detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangles around each detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue color, thickness 2

            # Convert the OpenCV frame (BGR) to QImage (RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit the processed frame to the UI
            self.processed_frame.emit(qt_image)

            # Add a small delay to avoid excessive CPU usage
            QThread.msleep(30) # Approximately 30ms delay

        # After the loop stops, release webcam resources
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait() # Wait until the thread finishes

# Main PyQt5 application class
class FaceDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detector Application")
        self.setGeometry(100, 100, 640, 480) # Window position and size

        self.init_ui()

        self.worker_thread = None
        self.is_running = False

    def init_ui(self):
        # Vertical layout to arrange widgets
        self.layout = QVBoxLayout()

        # Label to display the video feed
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Button to start/stop detection
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.toggle_detection)
        self.layout.addWidget(self.start_button)

        self.setLayout(self.layout)

    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if self.worker_thread is None or not self.worker_thread.isRunning():
            self.worker_thread = FaceDetectionWorker()
            self.worker_thread.processed_frame.connect(self.update_video_feed)
            self.worker_thread.error_message.connect(self.show_error_message)
            self.worker_thread.start()
            self.is_running = True
            self.start_button.setText("Stop Detection")
        else:
            QMessageBox.warning(self, "Warning", "Detection is already running.")

    def stop_detection(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.is_running = False
            self.start_button.setText("Start Detection")
            self.video_label.clear() # Clear video display
            self.video_label.setText("Video Feed Stopped")
        else:
            QMessageBox.warning(self, "Warning", "Detection is not running.")

    def update_video_feed(self, qt_image):
        # Update the QPixmap in QLabel with the new frame
        pixmap = QPixmap.fromImage(qt_image)
        # Scale the pixmap to fit the label (optional, but good for responsiveness)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)
        self.stop_detection() # Stop detection if an error occurs

    def closeEvent(self, event):
        # Ensure the thread is stopped when the application closes
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectorApp()
    window.show()
    sys.exit(app.exec_())
