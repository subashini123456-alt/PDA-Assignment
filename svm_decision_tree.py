from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y)

# ---------- SVM ----------
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)
print("SVM Accuracy:", round(accuracy_score(y_test, svm_pred), 4))

# ---------- Decision Tree ----------
dt = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=7)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", round(accuracy_score(y_test, dt_pred), 4))
print()
print(classification_report(y_test, dt_pred, target_names=wine.target_names))

plt.figure(figsize=(12, 7))
plot_tree(dt, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ConfusionMatrixDisplay.from_predictions(y_test, svm_pred, ax=ax[0],
                                        cmap="Blues", colorbar=False)
ax[0].set_title("SVM")
ConfusionMatrixDisplay.from_predictions(y_test, dt_pred, ax=ax[1],
                                        cmap="Greens", colorbar=False)
ax[1].set_title("Decision Tree")
plt.tight_layout()
plt.show()
