
import pickle
import numpy as np
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, AveragePooling2D


with open("data/images.txt", "rb") as fp:   # Unpickling
   X = pickle.load(fp)

with open("data/labels.txt", "rb") as fp:  # Unpickling
   y = pickle.load(fp)

print(len(X))
train_X = np.asarray(X[:1200])
train_y = np.array(y[:1200])
test_X = np.asarray(X[1200:])
test_y = np.array(y[1200:])


model = Sequential()
model.add(Conv2D(32, kernel_size=3, strides=2, activation='relu',padding='SAME', input_shape=(224, 224, 3)))
model.add(MaxPool2D(pool_size=(3, 3), strides=2,padding='VALID'))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='SAME'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=1, padding='VALID'))
model.add(Conv2D(128, kernel_size=3,strides=2, activation='relu', padding='SAME'))
model.add(Dropout(.5))
model.add(Conv2D(64, kernel_size=3,strides=1, activation='relu', padding='SAME'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(.25))
model.add(Dense(31,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("models/my_keras_model.h5", save_best_only=True)

history = model.fit(train_X, train_y, epochs=100, validation_split=0.2,batch_size=16,
                    callbacks=[early_stopping_cb, checkpoint_cb])

pred = model.predict(test_X)


print(np.sqrt(np.mean((pred - test_y)**2)))
