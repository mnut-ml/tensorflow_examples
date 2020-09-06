from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape( 60000, 28*28)
train_images = train_images.astype( 'float32')/255

test_images = test_images.reshape( 10000, 28*28)
test_images = test_images.astype( 'float32')/255

train_labels = to_categorical( train_labels)
test_labels = to_categorical( test_labels)


visible = Input( shape=(28*28,))
d1 = Dense( 512, activation  = 'relu')( visible )
d2 = Dense( 10, activation = 'softmax') (d1)

model = Model( input = visible, output = d2)

model.compile( optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])


nfold = 3
samples_per_fold = train_images.shape[0]//nfold
for n in range(0, nfold):
    partial_val_images = train_images[ n*samples_per_fold:(n+1)*samples_per_fold, :]
    partial_val_labels = train_labels[ n*samples_per_fold:(n+1)*samples_per_fold, :]

    partial_train_images = np.concatenate( np.array(partial_val_images), np.array(partial_val_images))
    #partial_train_labels = np.concatenate( [train_labels[:n*samples_per_fold,:]], [train_labels[(n+1)*samples_per_fold:,:]])

history = model.fit( x = train_images, y = train_labels, epochs=100, batch_size=128, validation_data=(test_images, test_labels))

d = history.history
l = d['loss']
v_l = d['val_loss']
e = range(1, len(l) + 1)

plt.plot(e, l, 'bo', label='Training loss')
plt.plot(e, v_l, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
