import cv2
import face_recognition
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def load_facebank(facebank_path):
    facebank = {}
    for filename in os.listdir(facebank_path):
        if filename.lower().endswith(".jpg"):
            name = filename.split("_faceid")[0]
            image_path = os.path.join(facebank_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                facebank[name] = encodings[0]
            else:
                print(f"No face found in {filename}")
    return facebank

# Load known faces from the facebank
facebank_path = "facebank"  # Replace with your facebank directory path
facebank = load_facebank(facebank_path)

# Open the RTSP stream 
rtsp_url = "rtsp://admin:L2F2A85E@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"  # Replace with your RTSP stream URL
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        break

    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute face encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Process each face found
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with known faces
        matches = face_recognition.compare_faces(list(facebank.values()), encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = list(facebank.keys())[first_match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Export the result
        result = {
            "name": name,
            "location": (left, top, right, bottom)
        }
        print(result)  # Replace with your desired export method (e.g., save to a file or database)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

