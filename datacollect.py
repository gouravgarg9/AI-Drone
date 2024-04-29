import cv2
import time
from deepface import DeepFace
from matplotlib import pyplot as plt
import os
import numpy as np
import random
import math
from sort import *
import pandas as pd
# Define directory paths
known_faces_dir = "data/people"  # Replace with your directory path
output_dir = "data/video"  # Replace with your directory path

# Load face recognition model (optional, choose a model from deepface.models)
model_name = "VGG-Face"  # Example model
# DeepFace.build_model(model_name)  # Uncomment if manual model loading is needed

# Initialize video capture
cap = cv2.VideoCapture(0)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


def is_valid(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    ##### Just for visualization and debug; remove in final
    # plt.plot(s)
    # plt.plot([p * 255, p * 255], [0, np.max(s)], 'r')
    # plt.text(p * 255 + 5, 0.9 * np.max(s), str(s_perc))
    # plt.show()
    ##### Just for visualization and debug; remove in final

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.7
    return s_perc > s_thr



def unknownFace(frame) :
    print("unkown face")
    if is_valid(frame) : 
	    cv2.imwrite(os.path.join(known_faces_dir, f"{random.random()}.jpg"), frame)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection (may improve accuracy)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = np.empty((0, 5))

    # Detect faces
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the ROI (region of interest) for the face
        roi = frame[y:y+h, x:x+w]

        # Recognize the face
        results = DeepFace.find(img_path=roi, db_path=known_faces_dir, model_name=model_name,enforce_detection = False)[0]
        # # Find the index of the face with the highest match confidence
        # print(results)
        if len(results) > 0 :   
            best_match_index = np.argmin(results["distance"])
            if results["distance"][best_match_index] < 0.4:
                name = f'{results["identity"][best_match_index]}'
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif results["distance"][best_match_index] < 0.8 :
                unknownFace(roi.copy());  # Threshold for confidence
            # Draw a rectangle around the face and label it with the recognized name or "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            dist = math.ceil((results["distance"][best_match_index] * 100)) / 100
            currentArray = np.array([x, y, (x+w), (y+h), dist])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    print(resultsTracker)

    # Save the processed frame with recognized face
    cv2.imwrite(os.path.join(output_dir, f"{time.time()}.jpg"), frame)

    # Display the resulting frame
    cv2.imshow('Real-time Face Recognition', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()
