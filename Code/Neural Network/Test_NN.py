import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import os

# Khai báo đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
test_fea_path = os.path.join(current_dir, "../../Data/test_fea.npy")
test_lab_path = os.path.join(current_dir, "../../Data/test_lab.npy")
model_path = os.path.join(current_dir, "NN.h5")

# Xử lý dữ liệu
test_features = np.load(test_fea_path, allow_pickle=True)
test_labels = np.load(test_lab_path, allow_pickle=True)
test_features = np.array(test_features, dtype=pd.Series)
test_labels = np.array(test_labels, dtype=pd.Series)
test_true = test_labels
test_class_label = test_labels
encoder = LabelEncoder()
encoder.fit(test_labels.astype(str))
encoded_Y = encoder.transform(test_labels.astype(str))
test_labels = to_categorical(encoded_Y)
test_labels.resize(test_labels.shape[0], 5)
ts_features = np.array(test_features, dtype=np.float32)

# Tải mô hình
model = load_model(model_path)

# Dự đoán nhãn dán
prediction = model.predict(ts_features)
test_predicted = []
labels_map = ["ANG", "FEA", "HAP", "NEU", "SAD"]
for i, val in enumerate(prediction):
    predicted_class = np.argmax(val) 
    test_predicted.append(labels_map[predicted_class])

# Hiển thị độ chính xác
print("Accuracy Score:", accuracy_score(test_true, test_predicted))
print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(test_labels))

# Vẽ ma trận nhầm lẫn
matrix = confusion_matrix(test_true, test_predicted)
classes = list(set(test_class_label))
classes.sort()
df = pd.DataFrame(matrix, columns=classes, index=classes)
sn.heatmap(
    df,
    annot=True,               
    fmt='.0f',                
    cmap='rocket_r',          
    square=True,              
    cbar_kws={"shrink": 0.75} 
)
plt.title('Confusion Matrix NN', fontsize=14)  
plt.xlabel('Predicted Label', fontsize=12) 
plt.ylabel('True Label', fontsize=12)      
plt.xticks(rotation=45)                    
plt.show()