import cv2
from cv2 import cv2 as Icv2
import os
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import face_recognition

tf.get_logger().setLevel('ERROR')

video_capture = cv2.VideoCapture(0)

model = models.load_model('weight/mask.model')
model.load_weights('weight/mask.h5')
probability_model = tf.keras.Sequential([model])

class_names = [" no mask", " mask", " bad mask position"]

face_locations = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)

    old_face_locations = face_locations
    face_locations = face_recognition.face_locations(frame)

    if not face_locations:
        face_locations = old_face_locations

    for face in face_locations:
        top, right, bottom, left = face
        roi = frame[top:bottom, left:right]
        try:
            output = []
            data = Icv2.resize(roi, (64,64))
            output.append(data)
            roi = np.array(output)
            roi = roi / 255.0
            predictions = probability_model.predict(roi)
            print(predictions)
            predict  = np.argmax(predictions)
            if predict== 1:
                cv2.rectangle(frame, (left, top), (right, bottom),(0,255,0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom),(0,255,0), -1)
            elif predict == 2 :
                cv2.rectangle(frame, (left, top), (right, bottom),(255,0,0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom),(255,0,0), -1)
            else :
                cv2.rectangle(frame, (left, top), (right, bottom),(0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom),(0,0,255), -1)
            cv2.putText(frame, class_names[predict], (left, bottom - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), 2)
        except Exception as e:
            pass

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
