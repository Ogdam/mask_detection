import cv2
from cv2 import cv2 as fff
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

tf.get_logger().setLevel('ERROR')

cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

model = models.load_model('./weight/mask.model')
model.load_weights('./weight/mask.h5')
probability_model = tf.keras.Sequential([model])

class_names = [" mask", " no mask", " bad position"]

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(64, 64),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for face in faces:
        (x,y,w,h) = face
        roi = frame[x:w, y:h]
        if len(roi) > 0 :
            try:
                output = []
                data = fff.resize(roi, (64,64))
                output.append(data)
                npRoi = np.array(output)
                npRoi = npRoi / 255.0
                predictions = probability_model.predict(npRoi)
                predict  = np.argmax(predictions)
                if predict == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
                    cv2.rectangle(frame, (x-1, y-25), (x+100, y),(0,255,0), -1)
                elif predict == 2 :
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(255,0,0), 2)
                    cv2.rectangle(frame, (x-1, y-25), (x+100, y),(255,0,0), -1)
                else :
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)
                    cv2.rectangle(frame, (x-1, y-25), (x+100, y),(0,0,255), -1)
                cv2.putText(frame, class_names[predict], (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), 2)
            except Exception as e:
                pass
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
