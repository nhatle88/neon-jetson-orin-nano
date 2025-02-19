import os
import cv2
import numpy as np
import face_recognition
import threading

# Global constants and shared variables
FACEBANK_PATH = "facebank"         # Folder with known face images
THRESHOLD = 0.5                   # Distance threshold for face recognition

# Shared dictionary for the latest processed frame from each stream
latest_frames = {}
frame_lock = threading.Lock()      # Lock to manage concurrent access to latest_frames


def align_face(face_image):
    """
    Aligns a face image so that the eyes are horizontal.
    Assumes face_image is in RGB format and cropped to the face region.
    """
    landmarks_list = face_recognition.face_landmarks(face_image)
    if not landmarks_list:
        return face_image  # Return original image if landmarks not detected

    landmarks = landmarks_list[0]
    left_eye_pts = landmarks["left_eye"]
    right_eye_pts = landmarks["right_eye"]
    left_eye_center = np.mean(left_eye_pts, axis=0).astype("int")
    right_eye_center = np.mean(right_eye_pts, axis=0).astype("int")

    # Calculate angle between the eye centers
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Calculate eyes center and cast to native Python ints
    eyes_center = (
        int((left_eye_center[0] + right_eye_center[0]) // 2),
        int((left_eye_center[1] + right_eye_center[1]) // 2)
    )

    # Rotate by negative angle to align eyes horizontally
    M = cv2.getRotationMatrix2D(eyes_center, -angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_image, M, (face_image.shape[1], face_image.shape[0]),
                                  flags=cv2.INTER_CUBIC)
    return aligned_face


def compute_confidence(distance, threshold=THRESHOLD, k=10):
    """
    Computes a confidence score (in percentage) from a face distance value using
    a logistic function. The confidence is near 100% when distance is low, and
    around 50% when distance equals the threshold.
    """
    confidence = 1 / (1 + np.exp(k * (distance - threshold))) * 100
    return confidence


class FaceRecognition:
    """Loads the facebank and performs face recognition with face alignment."""
    def __init__(self, facebank_path):
        self.facebank = self.load_facebank(facebank_path)

    def load_facebank(self, path):
        """Loads known face encodings from images stored in a folder (optionally align them)."""
        facebank = {}
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                base_name = filename.split("_faceid")[0].lower()
                name_map = {
                    "nhatle": "Nhat Le",
                    "thitran": "Thi Tran",
                    "hungle": "Hubert Le",
                    "stevenle": "Steven Le"
                }
                name = name_map.get(base_name, base_name)
                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                # Optionally, detect and align the face in the bank image
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_image = image[top:bottom, left:right]
                    aligned_face = align_face(face_image)
                    encodings = face_recognition.face_encodings(aligned_face)
                else:
                    encodings = face_recognition.face_encodings(image)

                if encodings:
                    facebank[name] = encodings[0]
                else:
                    print(f"[WARNING] No face found in {filename}")
        return facebank

    def recognize_faces(self, frame):
        """
        Detects faces in the frame, aligns each face for normalization,
        computes its encoding, and compares it with the facebank.
        Returns a list of tuples: (top, right, bottom, left, label).
        The label includes the recognized name and a confidence percentage.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        results = []

        for (top, right, bottom, left) in face_locations:
            face_image = rgb_frame[top:bottom, left:right]
            aligned_face = align_face(face_image)
            encodings = face_recognition.face_encodings(aligned_face)
            if not encodings:
                continue

            face_encoding = encodings[0]
            label = "Unknown"
            distances = [
                (bank_name, face_recognition.face_distance([bank_encoding], face_encoding)[0])
                for bank_name, bank_encoding in self.facebank.items()
            ]
            if distances:
                best_match, best_distance = min(distances, key=lambda x: x[1])
                if best_distance < THRESHOLD:
                    confidence = compute_confidence(best_distance)
                    label = f"{best_match} ({round(confidence, 1)}%)"
            results.append((top, right, bottom, left, label))
        return results


class RTSPStream:
    """Manages one RTSP stream and processes frames for face recognition."""
    def __init__(self, name, rtsp_url, face_recognizer):
        self.name = name
        self.rtsp_url = rtsp_url
        self.face_recognizer = face_recognizer
        self.video_capture = self.create_capture_pipeline(rtsp_url)
        self.running = True

    def create_capture_pipeline(self, rtsp_url):
        pipeline = (
            f"rtspsrc location={rtsp_url} latency=100 protocols=udp drop-on-latency=true ! "
            "rtph264depay ! h264parse ! queue max-size-buffers=1 ! nvv4l2decoder ! "
            "nvvidconv ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
        return capture

    def process_stream(self):
        """Continuously reads frames, performs face recognition, and updates the shared frame dictionary."""
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                print(f"[WARNING] Failed to grab frame from {self.rtsp_url}")
                continue

            faces = self.face_recognizer.recognize_faces(frame)
            for (top, right, bottom, left, label) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

            with frame_lock:
                latest_frames[self.name] = frame.copy()

        self.video_capture.release()


def main():
    rtsp_streams = {
        "Camera Labs": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
        "Camera Spaceship": "rtsp://admin:L297FC1C@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1"
    }

    face_recognizer = FaceRecognition(FACEBANK_PATH)
    streams = []
    threads = []

    for name, url in rtsp_streams.items():
        stream = RTSPStream(name, url, face_recognizer)
        streams.append(stream)
        thread = threading.Thread(target=stream.process_stream, daemon=True)
        threads.append(thread)
        thread.start()

    while True:
        with frame_lock:
            frames = [frame for frame in latest_frames.values() if frame is not None]

        if frames:
            try:
                combined_frame = np.hstack(frames)
                cv2.imshow("Combined Cameras", combined_frame)
            except Exception as e:
                print("Error combining frames:", e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for stream in streams:
        stream.running = False
    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
