# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pandas as pd
import os
import pickle
 
# Define the local path for the project files
path = './'
 
def pipeline_model(img_path):
    # Face detection model
    faceDetectionModel = os.path.join(path, 'Models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
 
    # Face detection model architecture
    faceDetectionProto = os.path.join(path, 'Models/deploy.prototxt.txt')
 
    # Face descriptor model
    faceDescriptor = os.path.join(path, 'Models/openface.nn4.small2.v1.t7')
 
    # Load the face detection model
    detectorModel = cv2.dnn.readNetFromCaffe(faceDetectionProto, faceDetectionModel)
 
    # Load the face descriptor model
    descriptorModel = cv2.dnn.readNetFromTorch(faceDescriptor)
 
    # Load the face recognition model
    face_recognition_model = pickle.load(open(os.path.join(path, 'ml_face_person_identity.pkl'), 'rb'))
 
    # Load the image
    img = cv2.imread(img_path)
    image = img.copy()
    h, w = img.shape[:2]
 
    # Face detection
    img_blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)
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
 
    if len(detections) > 0:
        for i, confidence in enumerate(detections[0, 0, :, 2]):
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startx, starty, endx, endy = box.astype(int)
 
                cv2.rectangle(image, (startx, starty), (endx, endy), (0, 255, 0), 2)
 
                # Feature extraction
                face_roi = img[starty:endy, startx:endx]
                face_blob = cv2.dnn.blobFromImage(face_roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
                descriptorModel.setInput(face_blob)
                vectors = descriptorModel.forward()
 
                # Predict with machine learning
                face_name = face_recognition_model.predict(vectors)[0]
 
                # Calculate face_name_score based on classifier agreement (e.g., for hard voting)
                individual_predictions = [clf.predict(vectors)[0] for clf in face_recognition_model.estimators_]
                agreement_count = sum([1 for pred in individual_predictions if pred == face_name])
                total_classifiers = len(face_recognition_model.estimators_)
                face_score = agreement_count / total_classifiers
 
                # Annotate image
                text_face = '{}'.format(face_name)
                cv2.putText(image, text_face, (startx, starty), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
 
                machinelearning_results['count'].append(count)
                machinelearning_results['face_detect_score'].append(confidence)
                machinelearning_results['face_name'].append(face_name)
                machinelearning_results['face_name_score'].append(face_score)
                count += 1
 
    return image, machinelearning_results
 
# Test the pipeline on sample images
image_path = os.path.join(path, 'face1.jpg')
img, results = pipeline_model(image_path)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()