# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 02:17:26 2018

@author: jaydeep thik
"""

"""support to AE"""

from sklearn.utils import shuffle
import keras
from keras import datasets
import matplotlib.pyplot as plt

id= shuffle([i for i in range(500, 600)])

model = keras.models.load_model('ae_model_20.h5')

(X_train, y_train),(X_test, y_test) = datasets.mnist.load_data()

latent_dim = 10

X_train = X_train.reshape((60000, 28,28,1))
X_test = X_test.reshape((10000, 28,28,1))


X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.


for idx in id:
    regen = model.predict(X_test[idx].reshape((1,28,28,1)))
    regen = regen*225
    regen = regen.astype('int32')
    regen =regen.reshape((28,28))
    
    print("recreated image")
    plt.imshow(regen, cmap=plt.cm.gray)
    plt.show()
    
    print("\n original Image ", idx)
    xx =  255*X_test[idx]
    plt.imshow(xx.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    ii = input("")
    if ii=='q':
        break