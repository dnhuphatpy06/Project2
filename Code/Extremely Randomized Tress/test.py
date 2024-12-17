import glob
import librosa
import librosa.display
import numpy as np
import _pickle as pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

ts_features = np.load(r"D:\Project2\Data\test_fea.npy", allow_pickle=True)
ts_labels = np.load(r"D:\Project2\Data\test_lab.npy", allow_pickle=True)

ts_features = np.array(ts_features, dtype=pd.Series)
ts_labels = np.array(ts_labels, dtype=pd.Series)

filename = r"D:\Project_2\ExtraTreeClassified\extratree.sav"

model = pickle.load(open(filename, 'rb'))

prediction = model.predict(ts_features)

test_true = ts_labels

test_predicted = []

for i, val in enumerate(prediction):
    test_predicted.append(val)

print('Accuracy Score:', accuracy_score(test_true, test_predicted))

print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(ts_labels))

matrix = confusion_matrix(test_true, test_predicted)
classes = list(set(ts_labels))
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