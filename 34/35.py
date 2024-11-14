# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pickle
import os
import time
 
# Define the local path for the project files
path = './'
 
# Load models and setup
faceDetectionModel = os.path.join(path, 'Models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
faceDetectionProto = os.path.join(path, 'Models/deploy.prototxt.txt')
faceDescriptor = os.path.join(path, 'Models/openface.nn4.small2.v1.t7')
face_recognition_model = pickle.load(open(os.path.join(path, 'ml_face_person_identity.pkl'), 'rb'))
 
# Load the face detection and descriptor models
detectorModel = cv2.dnn.readNetFromCaffe(faceDetectionProto, faceDetectionModel)
descriptorModel = cv2.dnn.readNetFromTorch(faceDescriptor)
 
def pipeline_model(frame):
    """Process a frame and return the annotated frame and recognition results."""
    image = frame.copy()
    h, w = frame.shape[:2]
    # Face detection
    img_blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)
    detectorModel.setInput(img_blob)
    detections = detectorModel.forward()
 
    # Machine learning results
    machinelearning_results = {
        'face_detect_score': [],
        'face_name': [],
        'face_name_score': [],
        'count': []
    }
    count = 1
 
    # Process each detected face
    for i, confidence in enumerate(detections[0, 0, :, 2]):
        if confidence > 0.5:
            # Get the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startx, starty, endx, endy = box.astype(int)
 
            cv2.rectangle(image, (startx, starty), (endx, endy), (0, 255, 0), 2)
 
            # Extract face ROI
            face_roi = frame[starty:endy, startx:endx]
            face_blob = cv2.dnn.blobFromImage(face_roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
            descriptorModel.setInput(face_blob)
            vectors = descriptorModel.forward()
 
            # Predict face identity
            face_name = face_recognition_model.predict(vectors)[0]
 
            # Calculate face_name_score based on classifier agreement (for hard voting)
            individual_predictions = [clf.predict(vectors)[0] for clf in face_recognition_model.estimators_]
            agreement_count = sum([1 for pred in individual_predictions if pred == face_name])
            total_classifiers = len(face_recognition_model.estimators_)
            face_score = agreement_count / total_classifiers
 
            # Display the predicted name and score on the image
            text_face = '{}'.format(face_name)
            cv2.putText(image, text_face, (startx, starty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
            # Save results in dictionary
            machinelearning_results['count'].append(count)
            machinelearning_results['face_detect_score'].append(confidence)
            machinelearning_results['face_name'].append(face_name)
            machinelearning_results['face_name_score'].append(face_score)
            count += 1
 
    return image, machinelearning_results
 
# Start video capture from the webcam
cap = cv2.VideoCapture(0)
 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Process the current frame
    annotated_frame, results = pipeline_model(frame)
 
    # Display the annotated frame
    cv2.imshow('Webcam Face Recognition', annotated_frame)
 
    # Press 'q' to quit the webcam stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    # sleep
    time.sleep(0.25)
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()