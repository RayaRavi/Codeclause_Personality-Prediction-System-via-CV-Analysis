import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset

df = pd.read_csv("personality_predict.csv")

# Preprocess the data
cv_data = df['Gender']
labels = df['Personality (class label)']

# Split the dataset into training and testing sets
cv_train, cv_test, labels_train, labels_test = train_test_split(cv_data, labels, test_size=0.2, random_state=100)

# Vectorize the CV data
vectorize = CountVectorizer()
cv_train_vectorized = vectorize.fit_transform(cv_train)
cv_test_vectorized = vectorize.transform(cv_test)

# Train the Support Vector Machine (SVM) classifier
classifier = SVC()
classifier.fit(cv_train_vectorized, labels_train)

# Make predictions on the test set
predictions = classifier.predict(cv_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy*100)
