import os
import cv2
import numpy as np
import face_recognition
import threading
import time
import math

os.environ["GST_DEBUG"] = "*:3"  # or "*:4" for more verbosity

def choose_best_cols(n, frame_width=640, frame_height=480, max_width=1920, max_height=1080):
    best_cols = 1
    best_area = 0
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        grid_width = cols * frame_width
        grid_height = rows * frame_height
        if grid_width <= max_width and grid_height <= max_height:
            area = grid_width * grid_height
            if area > best_area:
                best_area = area
                best_cols = cols
    return best_cols


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


def combine_frames_grid(frames, cols=2):
    """
    Combine a list of frames into a grid layout with the given number of columns.
    The frames are resized so that they all have the same height.
    
    Args:
        frames: List of frames (numpy arrays).
        cols: Number of columns in the grid.
        
    Returns:
        A single image with all frames arranged in a grid.
    """
    if not frames:
        return None

    # Resize frames to the same height.
    # Use the minimum height among the frames.
    min_height = min(frame.shape[0] for frame in frames)
    resized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = min_height / h
        new_w = int(w * scale)
        resized_frames.append(cv2.resize(frame, (new_w, min_height)))

    # Organize frames into rows.
    rows = []
    for i in range(0, len(resized_frames), cols):
        # Extract one row of frames.
        row_frames = resized_frames[i:i + cols]
        # If the row has fewer frames than the desired columns,
        # you might want to pad with black images.
        if len(row_frames) < cols:
            # Create a black image with same height and width equal to the first frame's width.
            black_frame = np.zeros_like(row_frames[0])
            row_frames.extend([black_frame] * (cols - len(row_frames)))
        row = np.hstack(row_frames)
        rows.append(row)

    # Stack all rows vertically.
    grid = np.vstack(rows)
    return grid


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
                name = name_map.get(base_name, base_name)
                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    facebank[name] = encodings[0]
                else:
                    print(f"[WARNING] No face found in {filename}")
        return facebank

    def recognize_faces(self, frame):
        """
        Detects faces in a frame and returns their locations and recognized names.
        The frame provided here can be a cropped ROI.
        """
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
    """
    Manages one RTSP stream and processes frames for face recognition,
    liveness check, and (optionally) a Region of Interest (ROI).
    """

    def __init__(self, name, rtsp_url, face_recognizer, roi=None):
        """
        roi: Optional tuple (x, y, width, height) specifying the ROI for face recognition.
        """
        self.name = name
        self.rtsp_url = rtsp_url
        self.face_recognizer = face_recognizer
        self.roi = roi  # For example, (0, 0, frame_width, 200)
        self.video_capture = self.create_capture_pipeline(rtsp_url)
        self.running = True
        self.prev_frame = None

    def create_capture_pipeline(self, rtsp_url):
        """
        Depending on the URL, use a different GStreamer pipeline.
        """
        #pipeline.set_state(Gst.State.NULL) # clean up pipeline
        # For HIK VISION cameras
        if "aircity2025" in rtsp_url:
            pipeline = (
                f"rtspsrc location={rtsp_url} latency=100 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! videoscale ! video/x-raw,format=BGR,width=320,height=240 ! "
                "appsink drop=true max-buffers=1 sync=false "
            )
        # For IMOU cameras 
        else:
            pipeline = (
                f"rtspsrc location={rtsp_url} latency=100 protocols=udp drop-on-latency=true ! "
                "rtph264depay ! h264parse ! queue max-size-buffers=1 ! nvv4l2decoder ! "
                "nvvidconv ! videoconvert ! video/x-raw,format=BGR,width=320,height=240 ! "
                "appsink drop=true max-buffers=1 sync=false"
            ) 
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
        return capture

    def process_stream(self):
        """
        Continuously reads frames, performs liveness check, and recognizes faces
        only in the defined ROI (if provided). Also overlays the camera name on the frame.
        """
        first_warning = True
        frame_counter = 0
        skip_factor = 29 # process one frame out of every 30
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                if first_warning:
                    print(f"[WARNING] Failed to grab frame from {self.rtsp_url}")
                    first_warning = False
                # Create a black frame if the stream is not running    
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                text = "Stream Not Available"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (250, 0, 0), font_thickness)
                
            frame_counter += 1
            if frame_counter % skip_factor != 0:
                continue

            # If ROI is defined, use that portion for face recognition.
            if self.roi is not None:
                (roi_x, roi_y, roi_w, roi_h) = self.roi
                # Draw the ROI rectangle on the full frame for visualization.
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
                roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                faces = self.face_recognizer.recognize_faces(roi_frame)
                # Adjust detected face coordinates back to the full frame.
                adjusted_faces = []
                for (top, right, bottom, left, label) in faces:
                    adjusted_faces.append((top + roi_y, right + roi_x, bottom + roi_y, left + roi_x, label))
                faces = adjusted_faces
            else:
                faces = self.face_recognizer.recognize_faces(frame)

            for (top, right, bottom, left, label) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

            # Overlay the camera name on the frame (top-left corner).
            cv2.putText(frame, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[self.name] = frame.copy()
            self.prev_frame = frame.copy()

        self.video_capture.release()


def main():
    # Define RTSP URLs for cameras (update with your credentials and IPs)
    rtsp_streams = {
	"Labs 1": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"Labs 2": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"Labs 3": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"Labs 4": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"Labs 5": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"Labs 6": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"HIK Vision 1": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/202",
	"HIK Vision 2": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/302",
	"HIK Vision 3": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/402",
	"HIK Vision 4": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/502",
	"HIK Vision 5": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/602",
	"HIK Vision 6": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/702",
	"HIK Vision 7": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/802",
	"HIK Vision 8": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/902",
	"HIK Vision 9": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/1002",
	"HIK Vision 10": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/1102",
	"HIK Vision 11": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/1201",
	"HIK Vision 12": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/1402",
	"Spaceship": "rtsp://admin:L297FC1C@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1",
	"UVK G1": "rtsp://admin:L268C6B7@d5030edfff7a.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1",
	"UVK P1": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"UVK P2": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"UVK P3": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"UVK P4": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"UVK P5": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"UVK P6": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"LPM Gate": "rtsp://admin:L26EF2FD@hongnhung12.duckdns.org:559/cam/realmonitor?channel=1&subtype=1",
	"LBB F4": "rtsp://admin:L201353B@hcr086zs3b5.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1"
    } # 30 cameras

    # Define a dictionary mapping camera names to ROI tuples.
    # For example, here we set an ROI for "Camera Labs" (top of the frame) and leave others as full frame.
    camera_rois = {
       # "Labs": (50, 100, 500, 350),
       # "Spaceship": (20, 250, 400, 200),
       # "HIK Vision 1": (10, 10, 600, 400)
    }

    face_recognizer = FaceRecognition(FACEBANK_PATH)
    streams = []
    threads = []
    for name, url in rtsp_streams.items():
        roi = camera_rois.get(name)
        stream = RTSPStream(name, url, face_recognizer, roi=roi)
        streams.append(stream)
        thread = threading.Thread(target=stream.process_stream, daemon=True)
        threads.append(thread)
        thread.start()

    while True:
        with frame_lock:
            frames = [frame for frame in latest_frames.values() if frame is not None]
        if frames:
            try:
                combined_frame = combine_frames_grid(frames, cols=6)
                combined_frame_gpu = cv2.UMat(combined_frame)
                scale_percent = 90
                width = int(combined_frame_gpu.get().shape[1]*scale_percent/100)
                height = int(combined_frame_gpu.get().shape[0]*scale_percent/100)
                dim = (width, height)
                
                resized_frame_gpu = cv2.resize(combined_frame_gpu, dim, interpolation=cv2.INTER_NEAREST)
                resized_frame = resized_frame_gpu.get()
                cv2.imshow("AI Security Camera Streams", resized_frame)
            except Exception as e:
                print("Error combining frames:", e)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #time.sleep(1) # delay for 1 second

    for stream in streams:
        stream.running = False
    for t in threads:
        t.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
