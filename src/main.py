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

class_names = ["mask", "no mask"]

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = color[x:w, y:h]

        if len(roi) > 0 :
            try:
                output = []
                data = fff.resize(roi, (32,32))
                output.append(data)
                npRoi = np.array(output)
                npRoi = npRoi / 255.0
                probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
                predictions = probability_model.predict(npRoi)
                predict  = np.argmax(predictions)

                if predict == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
                else :
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(255,0,0), 2)
                print(class_names[predict])

                cv2.imshow('Video', frame)
            except:
                pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
