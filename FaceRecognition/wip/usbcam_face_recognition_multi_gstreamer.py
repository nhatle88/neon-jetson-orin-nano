import cv2
import face_recognition
import os
import numpy as np
import dlib

print(dlib.DLIB_USE_CUDA) # Should output true
print(dlib.__version__) # Show version

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

# --- Load your facebank ---
facebank = {}
facebank_path = "facebank"  # Folder with images like "nhatle_faceid.jpg" etc.
for filename in os.listdir(facebank_path):
    if filename.lower().endswith(".jpg"):
        # Map filenames to proper display names.
        base = filename.split("_faceid")[0]
        if base.lower() == "nhatle":
            name = "Nhat Le"
        elif base.lower() == "thitran":
            name = "Thi Tran"
        elif base.lower() == "hungle":
            name = "Hubert Le"
        elif base.lower() == "stevenle":
            name = "Steven Le"
        else:
            name = base
        image_path = os.path.join(facebank_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            facebank[name] = encodings[0]
        else:
            print(f"No face found in {filename}")

# Recognition threshold for matching
threshold = 0.5

# --- Open two USB cameras (device indices 0 and 1) ---
# Optimized GStreamer pipeline for USB webcam
gst_pipeline0 = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=5/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"
gst_pipeline1 = "v4l2src device=/dev/video2 ! video/x-raw, format=YUY2, width=640, height=480, framerate=5/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"
# OpenCV VideoCapture
cap0 = cv2.VideoCapture(gst_pipeline0, cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(gst_pipeline1, cv2.CAP_GSTREAMER)

#cap0 = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture(2)

if not cap0.isOpened() or not cap1.isOpened():
    print("One or more cameras could not be opened. Please check your connections.")
    exit()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # Process camera 0
    if ret0:
        rgb_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        face_locations0 = face_recognition.face_locations(rgb_frame0, model="cnn")
        face_encodings0 = face_recognition.face_encodings(rgb_frame0, face_locations0)
        face_names0 = []
        for encoding in face_encodings0:
            # Compare the current face encoding to those in the facebank
            distances = face_recognition.face_distance(list(facebank.values()), encoding)
            matches = face_recognition.compare_faces(list(facebank.values()), encoding, tolerance=threshold)
            name = "Not Registered"
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    # Convert the distance to a confidence percentage
                    confidence = face_confidence(distances[best_match_index], face_match_threshold=threshold)
                    name = f"{list(facebank.keys())[best_match_index]} ({confidence}%)"
            face_names0.append(name)
        for (top, right, bottom, left), name in zip(face_locations0, face_names0):
            cv2.rectangle(frame0, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame0, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #cv2.imshow("Camera 0 - Face Recognition", frame0)

    # Process camera 1 similarly
    if ret1:
        rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        face_locations1 = face_recognition.face_locations(rgb_frame1, model="cnn")
        face_encodings1 = face_recognition.face_encodings(rgb_frame1, face_locations1)
        face_names1 = []
        for encoding in face_encodings1:
            distances = face_recognition.face_distance(list(facebank.values()), encoding)
            matches = face_recognition.compare_faces(list(facebank.values()), encoding, tolerance=threshold)
            name = "Not Registered"
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    confidence = face_confidence(distances[best_match_index], face_match_threshold=threshold)
                    name = f"{list(facebank.keys())[best_match_index]} ({confidence}%)"
            face_names1.append(name)
        for (top, right, bottom, left), name in zip(face_locations1, face_names1):
            cv2.rectangle(frame1, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame1, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #cv2.imshow("Camera 1 - Face Recognition", frame1)
        
        combined = np.hstack((frame0, frame1))
        cv2.imshow("Both Cameras", combined)


    # Exit on 'q'
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()

