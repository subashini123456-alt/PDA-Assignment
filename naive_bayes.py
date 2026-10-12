from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=1)

nb = GaussianNB(var_smoothing=1e-2)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Purples")
plt.title("Naive Bayes - Confusion Matrix (Digits)")
plt.show()

fig, axes = plt.subplots(2, 6, figsize=(9, 3.5))
for ax, img, p, t in zip(axes.ravel(), X_test, y_pred, y_test):
    ax.imshow(img.reshape(8, 8), cmap="gray_r")
    ax.set_title(f"P:{p} A:{t}", color="green" if p == t else "red")
    ax.axis("off")
plt.suptitle("Sample predictions (P = predicted, A = actual)")
plt.show()
