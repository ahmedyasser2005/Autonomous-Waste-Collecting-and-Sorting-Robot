from ultralytics import YOLO
from gpiozero import AngularServo
from time import sleep
import cv2

# Load the YOLO model
model = YOLO("exec/best_ncnn_model", task='detect')

# Setup servo with GPIO pin 18
# servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

# # Move servo to 90 degree
# servo.angle = 90
# sleep(2)
# servo.angle = None

# def lookLeft(coordinates):


# def lookRight(coordinates):

# def moveForward(coordinates):


# Open the video file
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model.predict(
            frame,
            device=0,
            imgsz=416,
            max_det=3,
            conf=0.25,
            iou=0.7
        )
        
        # Work Here!!!

        summary = results[0].summary(True)
        print(results[0].summary(True))


        # while :
        #     lookLeft()
        
        # while 
        #     moveForward()

        # # End

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()