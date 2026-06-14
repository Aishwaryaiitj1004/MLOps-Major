from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = fetch_olivetti_faces()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = joblib.load("savedmodel.pth")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Test Accuracy:", accuracy)