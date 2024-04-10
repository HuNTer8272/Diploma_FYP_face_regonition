import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
import os
import asyncio
import math
import sys
import face_recognition
import tensorflow as tf
import time as time

class FaceRecognition:
    def __init__(self):
        # Initialize Firebase
        cred = credentials.Certificate("db.json")
        firebase_admin.initialize_app(cred, {"databaseURL": "https://product-page-94cad-default-rtdb.asia-southeast1.firebasedatabase.app"})
        self.ref = db.reference('people')

        # Load known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('adesh'):
            face_images = cv2.imread(f'adesh/{image}')
            face_encoding = self.face_encodings(face_images)
            if face_encoding:
                self.known_face_encodings.append(face_encoding[0])
                self.known_face_names.append(os.path.splitext(image)[0])
            else:
                print(f"No face encoding found for {image}")

    async def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        # rtsp
        # video_capture = cv2.VideoCapture("rtsp://admin:L284AA25@192.168.40.218:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
        
        if not video_capture.isOpened():
            sys.exit("Video source not found...")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame from video stream")
                break

            # Downscale the frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            face_encodings = []
            for (x, y, w, h) in faces:
                # Extract face region and resize
                face_image = cv2.resize(small_frame[y:y + h, x:x + w], (150, 150))
                face_encoding = self.face_encodings(face_image)
                if face_encoding:
                    face_encodings.append(face_encoding[0])

            face_names = []
            for face_encoding in face_encodings:
                matches = self.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                confidence = None

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                    face_distances = self.face_distance(self.known_face_encodings, face_encoding)
                    confidence = face_confidence(face_distances[first_match_index])

                face_names.append((name, confidence))

            # Update status in Firebase
            detected_names = [name for name, _ in face_names]
            self.update_status(detected_names)

            # Display annotations
            for (x, y, w, h), (name, confidence) in zip(faces, face_names):
                # Calculate text width and center horizontally
                (text_width, text_height), _ = cv2.getTextSize(f"{name} ({confidence})", cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
                text_x = x * 4 + (w * 2 * 4 - text_width) // 2
                # Center text vertically
                text_y = (y + h - 6) * 4 + (text_height + 6) // 2

                cv2.rectangle(frame, (x * 4, y * 4), ((x + w) * 4, (y + h) * 4), (0, 0, 255), 2)
                cv2.rectangle(frame, (x * 4, (y + h - 6) * 4), ((x + w) * 4, (y + h) * 4), (0, 0, 255), -1)
                cv2.putText(frame, f"{name} ({confidence})", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                            (255, 255, 255), 1)

            cv2.imshow("Face Recognition", frame)

            frame_count += 1
            if frame_count % 500000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def update_status(self, detected_names):
        # Get data from the database
        data = self.ref.get()

        if data:
            # Update status based on detected names
            for person_id, person_data in data.items():
                name = person_data.get('name', '')
                status = 'present' if name in detected_names else 'away'
                self.ref.child(person_id).update({'status': status})
        else:
            print("No data available in the database.")

    def compare_faces(self, known_face_encodings, face_encoding, tolerance=0.4):
        return [self.compare_face(face_encoding, known_face_encoding, tolerance) for known_face_encoding in
                known_face_encodings]

    def compare_face(self, face_encoding1, face_encoding2, tolerance=0.4):
        return np.linalg.norm(face_encoding1 - face_encoding2) <= tolerance

    def face_distance(self, known_face_encodings, face_encoding):
        return [np.linalg.norm(face_encoding - known_face_encoding) for known_face_encoding in known_face_encodings]

    def face_encodings(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return face_recognition.face_encodings(rgb_image)

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.10 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"


if __name__ == "__main__":
    # Use GPU for TensorFlow operations
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    fr = FaceRecognition()
    asyncio.run(fr.run_recognition())
