from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import _pickle as pickle

#Đường dẫn:
val_fea = 'val_fea.npy'
val_lab = 'val_lab.npy'
path_to_npy = 'D:/Project2/Data'

filename = "D:\Project2\Code\Decision Tree\DecisionTree.sav"

ts_features = np.load(f'{path_to_npy}/{val_fea}', allow_pickle=True)
ts_labels = np.load(f'{path_to_npy}/{val_lab}', allow_pickle=True)

# Load saved model from file
model = pickle.load(open(filename, 'rb'))

# Predict the emotion class
prediction = model.predict(ts_features)

# True labels
test_true = ts_labels

# List to store prediction
test_predicted = []

# Iterate over model prediction and store it into list
for i, val in enumerate(prediction):
    test_predicted.append(val)


# Accuracy score of model
print('Accuracy Score of Decision Tree model is:', accuracy_score(test_true, test_predicted))

# Number of corrected prediction
print('Number of correct prediction by using Decision Tree is:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(ts_labels))

# Plotting confusion matrix
matrix = confusion_matrix(test_true, test_predicted)
classes = list(set(ts_labels))
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
plt.title('Confusion Matrix Decision Tree', fontsize=14)  
plt.xlabel('Predicted Label', fontsize=12) 
plt.ylabel('True Label', fontsize=12)      
plt.xticks(rotation=45)                    
plt.show()