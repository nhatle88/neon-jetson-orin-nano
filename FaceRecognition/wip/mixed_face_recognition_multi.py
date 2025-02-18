import os
import cv2
import numpy as np
import face_recognition


def face_confidence(face_distance, face_match_threshold=0.6):
    """
    Converts a face distance into a confidence percentage.
    A distance of 0 will yield 100% confidence and a distance equal to 
    the threshold will yield around 50% confidence.
    """
    if face_distance > face_match_threshold:
        # For distances above threshold, confidence falls off rapidly
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)
        confidence = linear_val * 100
    else:
        # For distances below the threshold, confidence is higher
        linear_val = 1.0 - (face_distance / (face_match_threshold * 2.0))
        confidence = linear_val * 100
    return round(confidence, 2)

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
rtsp_url2 = "rtsp://admin:L297FC1C@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1" # Camera Spaceship

pipeline1 = (
    "rtspsrc location=rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1 latency=0 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"
)

pipeline2 = (
    "rtspsrc location=rtsp://admin:L297FC1C@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1 latency=0 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"
)

# Open the RTSP stream using the GStreamer pipeline
#video_capture_1 = cv2.VideoCapture(rtsp_url1, cv2.CAP_FFMPEG)
#video_capture_1.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimal internal buffer
video_capture_1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)

#video_capture_2 = cv2.VideoCapture(rtsp_url2)
video_capture_2 = cv2.VideoCapture(pipeline2, cv2.CAP_GSTREAMER)

# --- Open two USB cameras (device indices 0 and 1) ---
#gst_pipeline_3 = "v4l2src device=/dev/video0 ! video/x-raw, width=1280, height=720, framerate=30/1 ! videoconvert ! appsink"
#gst_pipeline_4 = "v4l2src device=/dev/video2 ! video/x-raw, width=1280, height=720, framerate=30/1 ! videoconvert ! appsink"
#video_capture_3 = cv2.VideoCapture(gst_pipeline_3, cv2.CAP_GSTREAMER)
#video_capture_4 = cv2.VideoCapture(gst_pipeline_4, cv2.CAP_GSTREAMER)

video_capture_3 = cv2.VideoCapture(0)
video_capture_4 = cv2.VideoCapture(2)

if not video_capture_1.isOpened() or not video_capture_2.isOpened():
    print("Error: Unable to open RTSP stream with the specified pipeline.")
    exit()

# Define a threshold for face distance
threshold = 0.5

while True:
    ret1, frame1 = video_capture_1.read() # ip cam 1
    ret2, frame2 = video_capture_2.read() # ip cam 2
    ret3, frame3 = video_capture_3.read() # usb cam 1
    ret4, frame4 = video_capture_4.read() # usb cam 2

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
    # cv2.imshow("Face Recognition Cam 1", frame1)
    
    
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
    # cv2.imshow("Face Recognition Cam 2", frame2)
    
    # Process camera usb 0
    if ret3:
        rgb_frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        face_locations3 = face_recognition.face_locations(rgb_frame3, model="cnn")
        face_encodings3 = face_recognition.face_encodings(rgb_frame3, face_locations3)
        face_names3 = []
        for encoding in face_encodings3:
            # Compare the current face encoding to those in the facebank
            distances = face_recognition.face_distance(list(facebank.values()), encoding)
            matches = face_recognition.compare_faces(list(facebank.values()), encoding, tolerance=threshold)
            name = "Unknown"
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    # Convert the distance to a confidence percentage
                    confidence = face_confidence(distances[best_match_index], face_match_threshold=threshold)
                    name = f"{list(facebank.keys())[best_match_index]} ({confidence}%)"
            face_names3.append(name)
        for (top, right, bottom, left), name in zip(face_locations3, face_names3):
            cv2.rectangle(frame3, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame3, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #cv2.imshow("Camera 0 - Face Recognition", frame0)

    # Process camera usb 1 
    if ret4:
        rgb_frame4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
        face_locations4 = face_recognition.face_locations(rgb_frame4, model="cnn")
        face_encodings4 = face_recognition.face_encodings(rgb_frame4, face_locations1)
        face_names4 = []
        for encoding in face_encodings4:
            distances = face_recognition.face_distance(list(facebank.values()), encoding)
            matches = face_recognition.compare_faces(list(facebank.values()), encoding, tolerance=threshold)
            name = "Unknown"
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    confidence = face_confidence(distances[best_match_index], face_match_threshold=threshold)
                    name = f"{list(facebank.keys())[best_match_index]} ({confidence}%)"
            face_names4.append(name)
        for (top, right, bottom, left), name in zip(face_locations4, face_names4):
            cv2.rectangle(frame4, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame4, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    combined_usb = np.hstack((frame1, frame2))
    cv2.imshow("Both USB Cameras", combined_usb)
    combined_ip = np.hstack((frame3, frame4))
    cv2.imshow("Both USB Cameras", combined_ip)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture_1.release()
video_capture_2.release()
cv2.destroyAllWindows()
