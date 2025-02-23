from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
X, y, class_names = iris.data, iris.target, iris.target_names
k = 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

pred = []
for x_test in X_test:
    distances = [np.linalg.norm(x_test - x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    pred.append(most_common[0][0])
y_pred = np.array(pred)


print('Accuracy:', np.mean(y_pred == y_test))
print('Predictions:', class_names[y_pred])
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test, y_pred))
