import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def baseline_model():
    model = Sequential()
    model.add(Dense(256, input_dim=193, activation="relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.3))  
    model.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation="softmax", kernel_initializer="glorot_uniform"))  
    optimizer = Adam(learning_rate=0.0001)  
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


train_fea = np.load(r"D:\Project2\Data\train_fea.npy", allow_pickle=True)
train_lab = np.load(r"D:\Project2\Data\train_lab.npy", allow_pickle=True)

val_fea = np.load(r"D:\Project2\Data\val_fea.npy", allow_pickle=True)
val_lab = np.load(r"D:\Project2\Data\val_lab.npy", allow_pickle=True)

fea = np.vstack((train_fea, val_fea))
lab = np.hstack((train_lab, val_lab))

X = np.array(fea, dtype=np.float32)
Y = np.array(lab, dtype=str)

seed = 7
np.random.seed(seed)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print("Classes: ", encoder.classes_)
dummy_y = to_categorical(encoded_Y)


model = baseline_model()

result = model.fit(X, dummy_y, validation_split=1/9, batch_size=64, epochs=200, verbose=1)

print("Baseline: %.2f%% (%.2f%%)" % (result.history['accuracy'][-1] * 100, np.std(result.history['accuracy']) * 100))

filename = 'keras_model.h5'
model.save(filename)
print('Model Saved..')

plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('Keras Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Vẽ đồ thị Loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Keras Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
