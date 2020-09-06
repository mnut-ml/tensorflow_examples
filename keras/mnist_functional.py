from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

plt.ion()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape( 60000, 28*28)
train_images = train_images.astype( 'float32')/255

test_images = test_images.reshape( 10000, 28*28)
test_images = test_images.astype( 'float32')/255

train_labels = to_categorical( train_labels)
test_labels = to_categorical( test_labels)


visible = Input( shape=(28*28,))

use_logistic_regression = False
if use_logistic_regression:
    d1 = Dense( 512, activation  = 'relu')( visible )
    d2 = Dense( 10, activation = 'softmax') (d1)

    model = Model( input = visible, output = d2)
else:
    d1 = Dense( 10, activation = 'softmax') (visible)
    model = Model( input = visible, output = d1)

model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit( x = train_images, y = train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))

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
