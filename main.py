s# OPENCV MODEL

from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Lenovo\OneDrive\Desktop\UPLOAD\MIniProject\Actual_MiniProject\haar_cascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Lenovo\OneDrive\Desktop\UPLOAD\MIniProject\Actual_MiniProject\trainedModel.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        region_of_interest_gray = gray[y:y+h,x:x+w]
        region_of_interest_gray = cv2.resize(region_of_interest_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([region_of_interest_gray])!=0:
            region_of_interest = region_of_interest_gray.astype('float')/255.0
            region_of_interest = img_to_array(region_of_interest)
            region_of_interest = np.expand_dims(region_of_interest,axis=0)

            prediction = classifier.predict(region_of_interest)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x+40,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 165, 0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()