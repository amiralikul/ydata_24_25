import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

print(X)

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize kNN Classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Step 4: Train the Classifier
knn.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = knn.predict(X_test)
print('y-pred', len(y_pred))



# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with k={k}: {accuracy:.2f}")

# Step 7: Predict for a New Sample
new_sample = np.array([[5.0, 3.6, 1.4, 0.2]])  # Example: Sepal length, Sepal width, Petal length, Petal width
predicted_class = knn.predict(new_sample)
print(f"Predicted class for new sample: {iris.target_names[predicted_class][0]}")
