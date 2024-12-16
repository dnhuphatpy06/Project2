import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

def baseline_model():
    """
    Tạo và trả về một mô hình học sâu (deep learning) cơ bản với kiến trúc mạng neural fully connected 
    (dense neural network) cho bài toán phân loại nhiều lớp (multi-class classification).

    Kiến trúc mô hình bao gồm:
    - Một lớp đầu vào (input layer) với 193 đặc trưng (features), sử dụng 256 nút (neurons) và hàm kích hoạt ReLU.
    - Các lớp ẩn (hidden layers) với 128 và 64 nút, mỗi lớp sử dụng hàm kích hoạt ReLU và dropout để giảm thiểu overfitting.
    - Lớp đầu ra (output layer) với 5 nút, sử dụng hàm kích hoạt Softmax để phân loại đa lớp.
    - Mô hình sử dụng tối ưu hóa Adam với tốc độ học thấp (learning rate) là 0.0001.

    Mô hình được biên dịch với hàm mất mát categorical crossentropy và đo lường độ chính xác (accuracy) trong quá trình huấn luyện.

    Trả về:
        model (keras.Sequential): Mô hình học sâu đã được biên dịch sẵn, sẵn sàng cho việc huấn luyện.
    """
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

# Khai báo đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
train_fea_path = os.path.join(current_dir, "../../Data/train_fea.npy")
train_lab_path = os.path.join(current_dir, "../../Data/train_lab.npy")
val_fea_path = os.path.join(current_dir, "../../Data/val_fea.npy")
val_lab_path = os.path.join(current_dir, "../../Data/val_lab.npy")
model_path = os.path.join(current_dir, "NN.h5")

# Xử lý dữ liệu
train_features = np.load(train_fea_path, allow_pickle=True)
train_labels = np.load(train_lab_path, allow_pickle=True)
val_features = np.load(val_fea_path, allow_pickle=True)
val_labels = np.load(val_lab_path, allow_pickle=True)
features = np.vstack((train_features, val_features))
labels = np.hstack((train_labels, val_labels))
X = np.array(features, dtype=np.float32)
Y = np.array(labels, dtype=str)
seed = 7
np.random.seed(seed)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

# Khởi tạo mô hình
model = baseline_model()

# Huấn luyện và lưu mô hình
result = model.fit(X, dummy_y, validation_split=1/9, batch_size=64, epochs=200, verbose=1)
model.save(model_path)
print('Model Saved..')

# Vẽ đồ thị Accuracy
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
