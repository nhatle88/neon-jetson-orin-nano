import cv2
import pytesseract
# import Jetson.GPIO as GPIO
from ultralytics import YOLO
import time
import onnx
model = onnx.load("license_plate.onnx")
onnx.checker.check_model(model)
print("The model is valid!")


# Initialize relay (to control the barrier)
# RELAY_PIN = 16
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(RELAY_PIN, GPIO.OUT)
# GPIO.output(RELAY_PIN, GPIO.LOW)

# Load the YOLO model for license plate detection
model = YOLO("yolov8n.pt")

# List of valid license plates (registered residents)
registered_plates = {"59V2-88626", "570L1-58560", "59K1-74473"}
# Open the USB camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = frame[y1:y2, x1:x2]  # Crop the license plate region

            # Convert to grayscale for improved OCR
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            # Recognize characters from the license plate
            text = pytesseract.image_to_string(thresh, config="--psm 7")
            text = text.strip().replace(" ", "")  # Normalize the license plate string

            # Check if the license plate is in the list
            if text in registered_plates:
                print(f"Valid vehicle: {text} - Opening barrier!")
             #   GPIO.output(RELAY_PIN, GPIO.HIGH)  # Open the barrier
             #   time.sleep(5)                     # Keep barrier open for 5 seconds
             #   GPIO.output(RELAY_PIN, GPIO.LOW)   # Close the barrier
            else:
                print(f"Invalid vehicle: {text}")

    cv2.imshow("License Plate Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# GPIO.cleanup()

