from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

print("Loading dataset...")

data = fetch_olivetti_faces()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

joblib.dump(model, "savedmodel.pth")

print("Model Saved Successfully")