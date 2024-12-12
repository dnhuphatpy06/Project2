import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

ts_features = np.load(r"D:\Project2\Data\test_fea.npy", allow_pickle=True)
ts_labels = np.load(r"D:\Project2\Data\test_lab.npy", allow_pickle=True)

ts_features = np.array(ts_features, dtype=pd.Series)
ts_labels = np.array(ts_labels, dtype=pd.Series)

test_true = ts_labels
test_class_label = ts_labels

encoder = LabelEncoder()
encoder.fit(ts_labels.astype(str))
encoded_Y = encoder.transform(ts_labels.astype(str))

ts_labels = to_categorical(encoded_Y)

ts_labels.resize(ts_labels.shape[0], 5)

model = load_model(r"D:\Project_2\Keras\keras_model.h5")

ts_features = np.array(ts_features, dtype=np.float32)
prediction = model.predict(ts_features)

test_predicted = []

labels_map = ["ANG", "FEA", "HAP", "NEU", "SAD"]

for i, val in enumerate(prediction):
    predicted_class = np.argmax(val) 
    test_predicted.append(labels_map[predicted_class])

print("Accuracy Score:", accuracy_score(test_true, test_predicted))
print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(ts_labels))

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