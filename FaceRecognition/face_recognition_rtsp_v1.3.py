import os
import cv2
import numpy as np
import face_recognition
import threading

# Global constants and shared variables
FACEBANK_PATH = "facebank"
THRESHOLD = 0.5  # Face distance threshold for recognition

# Shared dictionary for the latest processed frame from each stream
latest_frames = {}
frame_lock = threading.Lock()  # Protects access to latest_frames

def compute_confidence(distance, threshold=THRESHOLD, k=10):
    """
    Computes a confidence score (in percentage) from a face distance value using
    a logistic function. The confidence is near 100% when distance is low, and
    around 50% when distance equals the threshold.
    """
    confidence = 1 / (1 + np.exp(k * (distance - threshold))) * 100
    return confidence
    

class FaceRecognition:
    """Handles loading the facebank and performing face recognition."""
    def __init__(self, facebank_path):
        self.facebank = self.load_facebank(facebank_path)

    def load_facebank(self, path):
        facebank = {}
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                base_name = filename.split("_faceid")[0].lower()
                name_map = {
                    "nhatle": "Nhat Le",
                    "thitran": "Thi Tran",
                    "hungle": "Hubert Le",
                    "stevenle": "Steven Le",
                    "capham": "Ca Pham"
                }
                name = name_map.get(base_name, base_name)  # Fallback if not in mapping

                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    facebank[name] = encodings[0]
                else:
                    print(f"[WARNING] No face found in {filename}")
        return facebank

    def recognize_faces(self, frame):
        """Detects faces in a frame and returns their locations and recognized names."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        results = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            label = "Not Registered"
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
    def __init__(self, name, rtsp_url, face_recognizer, roi=None):
        self.name = name
        self.rtsp_url = rtsp_url
        self.face_recognizer = face_recognizer
        self.roi = roi
        self.video_capture = self.create_capture_pipeline(rtsp_url)
        self.running = True

    def create_capture_pipeline(self, rtsp_url):
        """Create an OpenCV VideoCapture using a GStreamer pipeline."""
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

            # Process the frame for face recognition
            if self.roi is not None:
                (roi_x, roi_y, roi_w, roi_h) = self.roi
                # Draw the ROI rectangle on the full frame for visualization
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
                roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                faces = self.face_recognizer.recognize_faces(roi_frame)
                # Adjust the face coordinates back to the original frame
                adjusted_faces = []
                for (top, right, bottom, left, label) in faces:
                    adjusted_faces.append((top + roi_y, right + roi_x, bottom + roi_y, left + roi_x, label))
                faces = adjusted_faces
            else:
                faces = self.face_recognizer.recognize_faces(frame)


            for (top, right, bottom, left, name) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)
            
            # Overlay the camera name on the frame (e.g., top-left corner)
            cv2.putText(frame, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)


            # Update the shared latest_frames dictionary
            with frame_lock:
                latest_frames[self.name] = frame.copy()

        self.video_capture.release()


def main():
    # Define RTSP URLs for cameras (update with your credentials and IPs)
    rtsp_streams = {
        "Labs": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
        "Spaceship": "rtsp://admin:L297FC1C@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1",
    }
    

    # Define a dictionary mapping camera names to ROI tuples.
    # For example, here we set an ROI for "Camera Labs" (top of the frame) and leave others as full frame.
    camera_rois = {
        "Labs": (50,100,500,350),
        "Spaceship": (20,250,400,200),
    }

    # Initialize face recognition using the facebank
    face_recognizer = FaceRecognition(FACEBANK_PATH)

    # Start each RTSP stream in its own thread
    streams = []
    threads = []
    for name, url in rtsp_streams.items():
        roi = camera_rois.get(name)
        stream = RTSPStream(name, url, face_recognizer, roi=roi)
        streams.append(stream)
        thread = threading.Thread(target=stream.process_stream, daemon=True)
        threads.append(thread)
        thread.start()

    # Main loop to combine frames from all streams and display them in one window
    while True:
        with frame_lock:
            # Only use frames that have been received
            frames = [frame for frame in latest_frames.values() if frame is not None]

        if frames:
            try:
                # Combine frames horizontally (ensure they have the same height)
                combined_frame = np.hstack(frames)
                cv2.imshow("Combined Cameras", combined_frame)
            except Exception as e:
                print("Error combining frames:", e)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop all stream threads gracefully
    for stream in streams:
        stream.running = False
    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# test git pull