
from sklearn.ensemble import ExtraTreesClassifier
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import _pickle as pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

tr_features = np.load(r"D:\Project2\Data\train_fea.npy", allow_pickle=True)
tr_labels = np.load(r"D:\Project2\Data\train_lab.npy", allow_pickle=True)

tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

exported_pipeline = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=20, min_samples_leaf=5,
                                         min_samples_split=8,max_depth=12,  n_estimators=1000)

exported_pipeline.fit(tr_features, tr_labels)

train_sizes, train_scores, test_scores = learning_curve(exported_pipeline, tr_features, tr_labels, n_jobs=-1, cv=8,
                                                        train_sizes=np.linspace(.1, 1.0, 5), verbose=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Extra Tree Model")
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

filename = 'extratree.sav'

pickle.dump(exported_pipeline, open(filename, 'wb'), protocol=2)

print('Model Saved..')

