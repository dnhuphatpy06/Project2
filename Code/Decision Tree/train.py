import librosa
import librosa.display
import numpy as np
import _pickle as pickle
from sklearn import model_selection
from sklearn import tree

#Đường dẫn:
train_fea = 'train_fea.npy'
train_lab = 'train_lab.npy'
path_to_npy = 'D:/Project_TH_2/data'

tr_features = np.load(f'{path_to_npy}/{train_fea}', allow_pickle=True)
tr_labels = np.load(f'{path_to_npy}/{train_lab}', allow_pickle=True)

X=tr_features.astype(int)
y=tr_labels.astype(str)

clf = tree.DecisionTreeClassifier(
    class_weight="balanced",
    criterion='entropy', 
    max_depth=30, 
    random_state=42, 
    min_samples_split=30, 
    min_samples_leaf=20
)

clf = clf.fit(X, y)

filename = 'D:/Project_TH_2/DecisionTree.sav' # Đường dẫn lưu mô hình

pickle.dump(clf, open(filename, 'wb'), protocol=2)

print('Model Saved..')
print('Score:', clf.score(X=tr_features.astype(int), y=tr_labels.astype(str)))

# Learning curve plotting
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
seed = 7
kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle=True)
train_sizes, train_scores, test_scores = learning_curve(clf, tr_features, tr_labels, n_jobs=-1, cv=kfold,
                                                        train_sizes=np.linspace(.1, 1.0, 5), verbose=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Decision Tree Model")
plt.legend(loc="best")
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
plt.legend(loc="lower right")
plt.show()