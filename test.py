'''  
PyPower Projects  
Emotion Detection Using AI  
'''

# USAGE: python test.py  

from keras.models import load_model  
from keras.preprocessing.image import img_to_array  
import cv2  
import numpy as np  

# Load the face classifier and emotion detection model  
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')  
classifier = load_model('./Emotion_Detection.h5')  

# Define the class labels for the emotions  
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  

# Start video capture from the webcam  
cap = cv2.VideoCapture(0)  

while True:  
    # Grab a single frame of video  
    ret, frame = cap.read()  
    if not ret:  
        print("Failed to grab frame")  
        break  

    # Convert the frame to grayscale  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    # Detect faces in the frame  
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  

    for (x, y, w, h) in faces:  
        # Draw a rectangle around the detected face  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        
        # Extract the region of interest (ROI) for emotion detection  
        roi_gray = gray[y:y + h, x:x + w]  
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  

        if np.sum([roi_gray]) != 0:  
            roi = roi_gray.astype('float32') / 255.0  # Normalize  
            roi = np.expand_dims(roi, axis=-1)  # Add channel dimension (48, 48, 1)  
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension (1, 48, 48, 1)  

            # Ensure input shape matches model expectations
            if roi.shape == (1, 48, 48, 1):  
                # Make prediction  
                preds = classifier.predict(roi)[0]  
                label = class_labels[preds.argmax()]  

                print("\nPrediction probabilities:", preds)  
                print("\nPredicted label:", label)  

                # Display the label on the frame  
                label_position = (x, y)  
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  
            else:
                print("Error: Input shape mismatch. Received shape:", roi.shape)
        else:  
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  

    # Display the frame with the detected emotions  
    cv2.imshow('Emotion Detector', frame)  

    # Break the loop on 'q' key press  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# Release the video capture and close windows  
cap.release()  
cv2.destroyAllWindows()  
