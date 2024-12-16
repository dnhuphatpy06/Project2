import numpy as np
import _pickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import os

# Khai báo đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
train_fea_path = os.path.join(current_dir, "../../Data/train_fea.npy")
train_lab_path = os.path.join(current_dir, "../../Data/train_lab.npy")
model_path = os.path.join(current_dir, "KNN.sav")

# Xử lý dữ liệu
train_features = np.load(train_fea_path, allow_pickle=True)
train_labels = np.load(train_lab_path, allow_pickle=True)
X=train_features.astype(int)
y=train_labels.astype(str)

# Khởi tạo mô hình
neigh = KNeighborsClassifier(
    algorithm= 'auto', 
    n_neighbors= 7, 
    p= 1, 
    weights= 'uniform'
)

# Huấn luyện và lưu mô hình
neigh = neigh.fit(X, y)
pickle.dump(neigh, open(model_path, 'wb'), protocol=2)
print('Model Saved..')

# Vẽ đường cong học tập
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