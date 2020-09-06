from keras import layers
from keras import models

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/manju/code/ML/data/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

ntrain = 200
nval = 100
ntest = 100

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

if 0:
  train_features, train_labels = extract_features(train_dir, ntrain)
  validation_features, validation_labels = extract_features(validation_dir, nval)
  test_features, test_labels = extract_features(test_dir, ntest)
  np.save('data/cats_and_dogs_small/vgg_train_features', train_features)
  np.save('data/cats_and_dogs_small/vgg_train_labels', train_labels)
  np.save('data/cats_and_dogs_small/vgg_val_features', validation_features)
  np.save('data/cats_and_dogs_small/vgg_val_labels', validation_labels)
  np.save('data/cats_and_dogs_small/vgg_test_features', test_features)
  np.save('data/cats_and_dogs_small/vgg_test_labels', test_labels)

else :
  train_features = np.load('data/cats_and_dogs_small/vgg_train_features.npy')
  train_labels = np.load('data/cats_and_dogs_small/vgg_train_labels.npy')
  validation_features = np.load('data/cats_and_dogs_small/vgg_val_features.npy')
  validation_labels = np.load('data/cats_and_dogs_small/vgg_val_labels.npy')
  test_features = np.load('data/cats_and_dogs_small/vgg_test_features.npy')
  test_labels = np.load('data/cats_and_dogs_small/vgg_test_labels.npy')

train_features = np.reshape(train_features, (ntrain, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (nval, 4 * 4 * 512))
test_features = np.reshape(test_features, (ntest, 4 * 4 * 512))

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

