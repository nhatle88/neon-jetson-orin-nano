#!/usr/bin/env python3
import cv2
import numpy as np
import pytesseract
import time

# ----- Configuration -----
model_path = "license_plate.onnx"  # Path to your ONNX detection model
confidence_threshold = 0.5           # Minimum confidence to consider a detection valid

# ----- Load the ONNX Model -----
# This uses OpenCV's DNN module to load and run inference on the CPU.
net = cv2.dnn.readNetFromONNX(model_path)
# Force CPU backend (default OpenCV DNN runs on CPU unless specified otherwise)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ----- Initialize Video Capture -----
# Using the default camera (device 0). Adjust if necessary.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting video capture (press 'q' to exit).")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame if needed (here we use 640x480 for the network input)
    resized_frame = cv2.resize(frame, (640, 480))

    # ----- Prepare the Input Blob -----
    # Scale the image to [0, 1] and swap BGR to RGB if required by your model.
    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1.0/255.0, size=(640, 480), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # ----- Run Inference -----
    start_time = time.time()
    detections = net.forward()
    inference_time = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time:.2f} ms")

    # The expected detection output is assumed to be a 4D blob with shape [1, 1, N, 7]
    detections = np.squeeze(detections)  # Now shape is (N, 7)

    # ----- Process Detections -----
    for detection in detections:
        confidence = detection[2]
        if confidence > confidence_threshold:
            # Convert normalized coordinates to pixel values (for resized frame)
            x1 = int(detection[3] * 640)
            y1 = int(detection[4] * 480)
            x2 = int(detection[5] * 640)
            y2 = int(detection[6] * 480)

            # Scale the coordinates back to the original frame size
            h, w = frame.shape[:2]
            x1 = int(x1 * (w / 640))
            y1 = int(y1 * (h / 480))
            x2 = int(x2 * (w / 640))
            y2 = int(y2 * (h / 480))

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ----- Extract the Region of Interest (ROI) for OCR -----
            plate_roi = frame[y1:y2, x1:x2]
            if plate_roi.size == 0:
                continue

            # Optional: Convert ROI to grayscale and apply thresholding for better OCR accuracy
            gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            _, thresh_roi = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)

            # ----- Run OCR with pytesseract -----
            # psm 7 assumes a single text line; adjust config as needed.
            config = "--psm 7"
            plate_text = pytesseract.image_to_string(thresh_roi, config=config)
            plate_text = plate_text.strip()
            print("Detected Plate Text:", plate_text)

            # Overlay the recognized text on the frame
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ----- Display the Resulting Frame -----
    cv2.imshow("License Plate OCR (CPU Inference)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

