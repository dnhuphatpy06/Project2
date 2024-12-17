import numpy as np
import _pickle as pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import os

# Khai báo đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
test_fea_path = os.path.join(current_dir, "../../Data/test_fea.npy")
test_lab_path = os.path.join(current_dir, "../../Data/test_lab.npy")
model_path = os.path.join(current_dir, "DecisionTree.sav")

# Xử lý dữ liệu
test_features = np.load(test_fea_path, allow_pickle=True)
test_labels = np.load(test_lab_path, allow_pickle=True)
test_features = np.array(test_features, dtype=pd.Series)
test_labels = np.array(test_labels, dtype=pd.Series)

# Tải mô hình
model = pickle.load(open(model_path, 'rb'))

# Dự đoán lớp cảm xúc
prediction = model.predict(test_features)
test_true = test_labels
test_predicted = []
for i, val in enumerate(prediction):
    test_predicted.append(val)

# Hiển thị độ chính xác của mô hình
print('Accuracy Score:', accuracy_score(test_true, test_predicted))
print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(test_labels))

# Vẽ ma trận nhầm lẫn
matrix = confusion_matrix(test_true, test_predicted)
classes = list(set(test_labels))
classes.sort()
df = pd.DataFrame(matrix, columns=classes, index=classes)
plt.figure(figsize=(8, 6))  
sn.heatmap(
    df,
    annot=True,               
    fmt='.0f',                
    cmap='rocket_r',          
    square=True,              
    cbar_kws={"shrink": 0.75} 
)
plt.title('Confusion Matrix ExtraTree', fontsize=14)  
plt.xlabel('Predicted Label', fontsize=12) 
plt.ylabel('True Label', fontsize=12)      
plt.xticks(rotation=45)                    
plt.show()