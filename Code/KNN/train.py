import librosa
import librosa.display
import numpy as np
import _pickle as pickle
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

"""Quá trình chạy file này tốn rất nhiều bộ nhớ và Ram, cân nhắc trước khi chạy"""

train_fea = 'train_fea.npy'
train_lab = 'train_lab.npy'
path_to_npy = 'D:/Project_TH_2/data'

tr_features = np.load(f'{path_to_npy}/{train_fea}', allow_pickle=True)
tr_labels = np.load(f'{path_to_npy}/{train_lab}', allow_pickle=True)

X=tr_features.astype(int)
y=tr_labels.astype(str)

neigh = KNeighborsClassifier(
    algorithm= 'auto', 
    n_neighbors= 7, 
    p= 1, 
    weights= 'uniform'
)

neigh = neigh.fit(X, y)

filename = 'D:/Project_TH_2/ModelKNN.sav'

pickle.dump(neigh, open(filename, 'wb'), protocol=2)

print('Model Saved..')
print('Score:', neigh.score(X=tr_features.astype(int), y=tr_labels.astype(str)))

# Learning curve plotting
seed = 7
kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle=True)
train_sizes, train_scores, test_scores = learning_curve(neigh, X, y, n_jobs=-1, cv=kfold,
                                                        train_sizes=np.linspace(.1, 1.0, 5), verbose=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("KNN Model")

plt.xlabel("Training examples")
plt.ylabel("Score")
plt.gca().invert_yaxis()

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")
line_up = plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
line_down = plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.ylim(-.1, 1.1)
plt.legend(loc="best")
plt.legend(loc="lower right")
plt.show()