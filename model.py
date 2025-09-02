import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model to a file named 'iris_model.pkl'
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as iris_model.pkl")