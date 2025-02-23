import os
import cv2
import numpy as np
import face_recognition
import threading

# Global shared variables
latest_frames = {}
frame_lock = threading.Lock()  # Protects access to latest_frames

def combine_frames_grid(frames, cols=2):
    if not frames:
        return None

    min_height = min(frame.shape[0] for frame in frames)
    resized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = min_height / h
        new_w = int(w * scale)
        resized_frames.append(cv2.resize(frame, (new_w, min_height)))

    rows = []
    for i in range(0, len(resized_frames), cols):
        row_frames = resized_frames[i:i + cols]
        if len(row_frames) < cols:
            black_frame = np.zeros_like(row_frames[0])
            row_frames.extend([black_frame] * (cols - len(row_frames)))
        row = np.hstack(row_frames)
        rows.append(row)

    grid = np.vstack(rows)
    return grid

class FaceDetection:
    def __init__(self, model="cnn"):
        self.model = model

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        return len(face_locations)

class RTSPStream:
    def __init__(self, name, rtsp_url, face_detector, roi=None):
        self.name = name
        self.rtsp_url = rtsp_url
        self.face_detector = face_detector
        self.roi = roi
        self.video_capture = self.create_capture_pipeline(rtsp_url)
        self.running = True

    def create_capture_pipeline(self, rtsp_url):
        if "aircity2025" in rtsp_url:
            pipeline = (
                f"rtspsrc location={rtsp_url} latency=100 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR,width=640,height=480 ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
        else:
            pipeline = (
                f"rtspsrc location={rtsp_url} latency=100 protocols=udp drop-on-latency=true ! "
                "rtph264depay ! h264parse ! queue max-size-buffers=1 ! nvv4l2decoder ! "
                "nvvidconv ! videoconvert ! video/x-raw,format=BGR,width=640,height=480 ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
        return cap

    def process_stream(self):
        first_warning = True
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                if first_warning:
                    print(f"[WARNING] Failed to grab frame from {self.rtsp_url}")
                    first_warning = False
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                text = "Stream Not Available"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 250), thickness)
            else:
                first_warning = True

            if self.roi is not None:
                (roi_x, roi_y, roi_w, roi_h) = self.roi
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
                roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                face_count = self.face_detector.detect_faces(roi_frame)
            else:
                face_count = self.face_detector.detect_faces(frame)

            cv2.putText(frame, f"{self.name} - Faces: {face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[self.name] = frame.copy()

        self.video_capture.release()

def main():
    rtsp_streams = {
	"Labs": "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
	"Spaceship": "rtsp://admin:L297FC1C@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1",
	"HIK Vision 1": "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/202",
	"UVK Parking": "rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1",
	"UVK Gate": "rtsp://admin:L268C6B7@d5030edfff7a.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1",
	"LBB F4": "rtsp://admin:L201353B@hcr086zs3b5.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1"
    }

    face_detector = FaceDetection(model="cnn")
    streams = []
    threads = []

    for name, url in rtsp_streams.items():
        stream = RTSPStream(name, url, face_detector)
        streams.append(stream)
        t = threading.Thread(target=stream.process_stream, daemon=True)
        threads.append(t)
        t.start()

    while True:
        with frame_lock:
            frames = [frame for frame in latest_frames.values() if frame is not None]
        if frames:
            try:
                combined_frame = combine_frames_grid(frames, cols=3)
                if combined_frame is not None:
                    cv2.imshow("AI Security Camera Streams", combined_frame)
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
