
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
import _pickle as pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
import os

# Khai báo đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
train_fea_path = os.path.join(current_dir, "../../Data/train_fea.npy")
train_lab_path = os.path.join(current_dir, "../../Data/train_lab.npy")
model_path = os.path.join(current_dir, "ExtraTree.sav")

# Xử lý dữ liệu
train_features = np.load(train_fea_path, allow_pickle=True)
train_labels = np.load(train_lab_path, allow_pickle=True)
train_features = np.array(train_features, dtype=pd.Series)
train_labels = np.array(train_labels, dtype=pd.Series)

# Khởi tạo mô hình
exported_pipeline = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=20, min_samples_leaf=5,
                                         min_samples_split=8,max_depth=12,  n_estimators=1000)

# Huấn luyện và lưu mô hình
exported_pipeline.fit(train_features, train_labels)
pickle.dump(exported_pipeline, open(model_path, 'wb'), protocol=2)
print('Model Saved..')

# Vẽ đường cong học tập
train_sizes, train_scores, test_scores = learning_curve(exported_pipeline, train_features, train_labels, n_jobs=-1, cv=8,
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

