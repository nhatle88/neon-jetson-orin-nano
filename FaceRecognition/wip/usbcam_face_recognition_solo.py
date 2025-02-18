import os
import cv2
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
video_capture = cv2.VideoCapture(0)

# Define a threshold for face distance
threshold = 0.5

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV default) to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        # Compare against each face in the facebank
        distances = []
        for bank_name, bank_encoding in facebank.items():
            # Compute the Euclidean distance between the embeddings
            dist = face_recognition.face_distance([bank_encoding], face_encoding)[0]
            distances.append((bank_name, dist))

        # Get the best match from the facebank
        if distances:
            best_match, best_distance = min(distances, key=lambda x: x[1])
            if best_distance < threshold:
                name = best_match

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Put the name label above the face rectangle
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
