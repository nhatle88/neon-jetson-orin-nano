import os
import cv2
import numpy as np
import face_recognition

# Path to the folder with facebank images
facebank_path = "facebank"

# Create a dictionary to hold names and their corresponding face encodings
facebank = {}

# Loop over each image in the facebank folder
for filename in os.listdir(facebank_path):
    if filename.endswith(".jpg"):
        # Here we assume filenames are like 'nhatle_faceid.jpg'
        base_name = filename.split("_faceid")[0]
        # Map the lowercase name to the desired display name
        if base_name.lower() == "nhatle":
            name = "Nhat Le"
        elif base_name.lower() == "thitran":
            name = "Thi Tran"
        elif base_name.lower() == "hungle":
            name = "Hubert Le"
        elif base_name.lower() == "stevenle":
            name = "Steven Le"
        else:
            name = base_name  # fallback

        # Load the image and compute the face encoding
        image_path = os.path.join(facebank_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            facebank[name] = encodings[0]
        else:
            print(f"No face found in {filename}")

# Open the video capture (USB camera)
# RTSP stream URL (adjust with your camera credentials and IP)
rtsp_url1 = "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1" # Camera Labs
rtsp_url2 = "rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/202" # Camera Spaceship

pipeline1 = (
    "rtspsrc location=rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1 latency=100 protocols=udp drop-on-latency=true ! "
    "rtph264depay ! h264parse ! queue max-size-buffers=1 ! nvv4l2decoder ! "
   # "nvvideoconvert ! video/x-raw(memory:NVMM),format=BGRx,width=640,height=480 ! "
   # "nvvidconv ! video/x-raw,format=BGR ! "
    "nvvidconv ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false "
)

pipeline2 = (
    "rtspsrc location=rtsp://admin:aircity2025@192.168.1.2:554/Streaming/channels/202 latency=100 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false "
)

# Open the RTSP stream using the GStreamer pipeline
#video_capture_1 = cv2.VideoCapture(rtsp_url1, cv2.CAP_FFMPEG)
#video_capture_1.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimal internal buffer
video_capture_1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)

#video_capture_2 = cv2.VideoCapture(rtsp_url2)
video_capture_2 = cv2.VideoCapture(pipeline2, cv2.CAP_GSTREAMER)
if not video_capture_1.isOpened() or not video_capture_2.isOpened():
    print("Error: Unable to open RTSP stream with the specified pipeline.")
    exit()


cap = video_capture_1
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Resolution: {width}x{height}, FPS: {fps}")

# Define a threshold for face distance
threshold = 0.5

while True:
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    # run the rtsp 1
    # Convert frame from BGR (OpenCV default) to RGB (face_recognition uses RGB)
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations1 = face_recognition.face_locations(rgb_frame1, model ="cnn")
    face_encodings1 = face_recognition.face_encodings(rgb_frame1, face_locations1)

    for (top, right, bottom, left), face_encoding1 in zip(face_locations1, face_encodings1):
        name = "Unknown"

        # Compare against each face in the facebank
        distances = []
        for bank_name, bank_encoding in facebank.items():
            # Compute the Euclidean distance between the embeddings
            dist = face_recognition.face_distance([bank_encoding], face_encoding1)[0]
            distances.append((bank_name, dist))

        # Get the best match from the facebank
        if distances:
            best_match, best_distance = min(distances, key=lambda x: x[1])
            if best_distance < threshold:
                name = best_match

        # Draw a rectangle around the face
        cv2.rectangle(frame1, (left, top), (right, bottom), (0, 255, 0), 2)
        # Put the name label above the face rectangle
        cv2.putText(frame1, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition Cam 1", frame1)
    
    
    # run the rtsp 2
    # Convert frame from BGR (OpenCV default) to RGB (face_recognition uses RGB)
    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations2 = face_recognition.face_locations(rgb_frame2, model ="cnn")
    face_encodings2 = face_recognition.face_encodings(rgb_frame2, face_locations2)

    for (top, right, bottom, left), face_encoding2 in zip(face_locations2, face_encodings2):
        name = "Unknown"

        # Compare against each face in the facebank
        distances = []
        for bank_name, bank_encoding in facebank.items():
            # Compute the Euclidean distance between the embeddings
            dist = face_recognition.face_distance([bank_encoding], face_encoding2)[0]
            distances.append((bank_name, dist))

        # Get the best match from the facebank
        if distances:
            best_match, best_distance = min(distances, key=lambda x: x[1])
            if best_distance < threshold:
                name = best_match

        # Draw a rectangle around the face
        cv2.rectangle(frame2, (left, top), (right, bottom), (0, 255, 0), 2)
        # Put the name label above the face rectangle
        cv2.putText(frame2, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition Cam 2", frame2)
    
   # combined = np.hstack((frame1, frame2))
   # cv2.imshow("Both Cameras", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture_1.release()
video_capture_2.release()
cv2.destroyAllWindows()
