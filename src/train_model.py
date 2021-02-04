# TensorFlow and tf.keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import datasets, layers, models
import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from cv2 import cv2
import random
import png

# ---------------------------------------- load_images -------------------------

train  = []
test = []

FOLDER_TRAIN = './dataset/train'
FOLDER_TEST = './dataset/test'

for file in os.listdir(FOLDER_TRAIN):
    src = cv2.imread(FOLDER_TRAIN+"/"+file)
    output = cv2.resize(src, (64,64))
    if "no_mask" in file:
        train.append((output, 0))
    elif "bad_mask" in file:
        train.append((output, 2))
    else:
        train.append((output, 1))

random.shuffle(train)
train_images, train_labels = zip(*train)

for file in os.listdir(FOLDER_TEST):
    src = cv2.imread(FOLDER_TEST+"/"+file)
    output = cv2.resize(src, (64,64))
    if "no_mask" in file:
        test.append((output, 0))
    elif "bad_mask" in file:
        test.append((output, 2))
    else:
        test.append((output, 1))

random.shuffle(test)
test_images, test_labels = zip(*test)

class_names = ["no mask", "good_mask", "bad mask position"]

train_labels = np.array(train_labels)
train_images = np.array(train_images)
train_images = train_images / 255.0

test_labels = np.array(test_labels)
test_images = np.array(test_images)
test_images = test_images / 255.0


# -------------------------------- create model --------------------------------
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(3))

#Compilez le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])

# model.load_weights('weight/mask.h5')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Former le modèle
history = model.fit(train_images, train_labels, epochs=5,
              validation_data=(test_images, test_labels),
              callbacks=[tensorboard_callback])

#Évaluer la précision
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

model.save_weights('./weight/mask.h5')
model.save('./weight/mask.model')

# add false positif to train set:
# for i in range(0, len(test_images)-1):
#     true_label, img = test_labels[i], test_images[i]
#     predicted_label = np.argmax(predictions[i])
#     if predicted_label != true_label :
#         newName = '{}.{}.jpg'.format(len(os.listdir(FOLDER_TEST)), class_names[true_label])
#         os.rename(FOLDER_TEST+"/"+os.listdir(FOLDER_TEST)[i], FOLDER_TRAIN+"/"+newName)
#         i = i-1


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
for i in range(0,12):
    plt.subplot(4,6,indexPlot)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(4,6,indexPlot+1)
    plot_value_array(i, predictions[i],  test_labels)
    indexPlot +=2
plt.savefig('./plt/test.png')
plt.show()

# Get training and test loss histories
training_acc = history.history['acc']
test_acc = history.history['val_acc']

training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)
# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.plot(epoch_count, training_acc, 'g--')
plt.plot(epoch_count, test_acc, 'y-')
plt.legend(['Training Loss', 'Test Loss', 'Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./plt/loss_plt.png')
plt.show()
