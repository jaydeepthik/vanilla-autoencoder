# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 00:51:11 2018

@author: jaydeep thik
"""

from keras import datasets, models, layers, optimizers
import matplotlib.pyplot as plt

(X_train, y_train),(X_test, y_test) = datasets.mnist.load_data()

latent_dim = 10

X_train = X_train.reshape((60000, 28,28,1))
X_test = X_test.reshape((10000, 28,28,1))


X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

X_tr = X_train[:50000]
X_valid = X_train[50000:]

#encoder model
model = models.Sequential()
model.add(layers.Conv2D(16,(3,3) , padding="SAME", input_shape=(28,28,1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D( 32   ,(3,3) , padding="SAME"))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D( 32   ,(3,3) , padding="SAME"))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))

#bottleneck
model.add(layers.Dense(latent_dim, activation='relu'))

#decoder model
model.add(layers.Dense(32*7*7, activation='relu'))
model.add(layers.Reshape((7,7,32)))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same'))

model.compile(optimizer=optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
summary = model.fit(X_tr, X_tr, validation_data=(X_valid, X_valid), batch_size=128, epochs=20)
model.save('ae_model_200.h5')

#visualize results

idx=36

regen = model.predict(X_tr[idx].reshape((1,28,28,1)))
regen = regen*225
regen = regen.astype('int32')
regen =regen.reshape((28,28))

print("recreated image")
plt.imshow(regen, cmap=plt.cm.gray)
plt.show()

print("\n original Image")
xx =  255*X_tr[idx]
plt.imshow(xx.reshape((28,28)), cmap=plt.cm.gray)
plt.show()