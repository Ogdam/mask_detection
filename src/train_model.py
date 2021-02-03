# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from cv2 import cv2

# load_images
train_images = []
train_labels = []
test_images = []
test_labels = []

FOLDER_TRAIN = './dataset/train'
FOLDER_TEST = './dataset/test/'

for file in os.listdir(FOLDER_TRAIN):
    src = cv2.imread(FOLDER_TRAIN+"/"+file)
    output = cv2.resize(src, (64,64))
    train_images.append(output)
    if "no_mask" in file:
        train_labels.append(1)
    elif "bad_mask" in file:
        train_labels.append(2)
    else:
        train_labels.append(0)

for file in os.listdir(FOLDER_TEST):
    src = cv2.imread(FOLDER_TEST+"/"+file)
    output = cv2.resize(src, (64,64))
    test_images.append(output)
    if "no_mask" in file:
        test_labels.append(1)
    elif "bad_mask" in file:
        test_labels.append(2)
    else:
        test_labels.append(0)

class_names = ["mask", "no mask", "bad position"]

train_labels = np.array(train_labels)
train_images = np.array(train_images)
train_images = train_images / 255.0

test_labels = np.array(test_labels)
test_images = np.array(test_images)
test_images = test_images / 255.0

print(len(test_images[0]))
print(len(test_images[0][0]))
print(test_images[0][0])

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#Construisez le modèle

model = models.Sequential()
model.add(layers.Flatten(input_shape=train_images.shape[1:]))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(3))

#Compilez le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Former le modèle
model.fit(train_images, train_labels, epochs=7,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])

#Évaluer la précision
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

model.save_weights('./weight/mask.h5')
model.save('./weight/mask.model')


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                64*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(3))
  plt.yticks([])
  thisplot = plt.bar(range(3), predictions_array, color="#777777")
  plt.ylim([0, 2])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


plt.figure()
indexPlot=1
for i in range(0,8):
    plt.subplot(4,6,indexPlot)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(4,6,indexPlot+1)
    plot_value_array(i, predictions[i],  test_labels)
    indexPlot +=2
plt.show()
