#Author: Sara Mazaheri
#Date: 2025-02-24
#Description: This is the first assignment of the course "NLP"

# Important Note:
# To evaluate train/validation/test accuracy, run line 120-128
# To evaluate stemed accuracy, run line 173-175
# To evaluate n-gram accuracy, run line 224-226
# To evaluate Logistic Regression best parameters, run line 274 and 282
# To evaluate SVM best parameters, run line 294 and 295
# To evaluate Naive Bayes best parameters, run line 305 and 306
# To evaluate Logistice Regression, SVM, and Naive Bayes confusion matrix, run line 336-337, 357,358 and 378-379 respectively.
# To evaluate Logistice Regression, SVM, and Naive Bayes evaluation metrics, run line 339-344, 360-365, and 3881-385 respectively.

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Define stopwords list
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv("amazon_reviews.csv")

# Drop missing values in reviews.text and reviews.rating
df = df.dropna(subset=["reviews.text", "reviews.rating"])

# Display basic info after cleaning
# print(df.info())
# print(df.head())

# Keep only the relevant columns
df = df[['reviews.text', 'reviews.rating']]

# Rename columns for easier access
df = df.rename(columns={'reviews.text': 'text', 'reviews.rating': 'rating'})

# Display dataset info and sample rows
# print(df.info())
# print(df.sample(5))

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Tokenize text
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)  # Join words back into a sentence

# Apply preprocessing to all reviews
df['text'] = df['text'].apply(preprocess_text)

# Show sample preprocessed reviews
# print(df.sample(5))

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Transform text into numerical features
X = vectorizer.fit_transform(df['text'])

# Convert to a DataFrame for better visualization
X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Display shape of transformed data
# print("Shape of feature matrix before using n-grams:", X_df.shape)

# Show a sample of the transformed text data
# print(X_df.sample(5))

# Define features (X) and labels (y)
y = df['rating']  # Target variable (star rating)

# Train/Validation/Test Split (60%/20%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Show dataset sizes
# print(f"Training set size: {X_train.shape}")
# print(f"Testing set size: {X_test.shape}")
# print(f"Validation set size: {X_val.shape}")

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_log_reg_val = log_reg.predict(X_val)


# Initialize SVM with a linear kernel
svm_model = SVC(kernel='linear', class_weight='balanced', C=0.1, random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)
y_pred_svm_val = svm_model.predict(X_val)


# Initialize Naive Bayes model
nb_model = MultinomialNB(alpha=0.2)

# Train the model
nb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_nb = nb_model.predict(X_test)
y_pred_nb_val = nb_model.predict(X_val)

# results = {
#     "Model": ["Logistic Regression","SVM", "Naivee Bayes"],
#     "Training Accuracy": [accuracy_score(y_train, log_reg.predict(X_train)), accuracy_score(y_train, svm_model.predict(X_train)), accuracy_score(y_train, nb_model.predict(X_train))],
#     "Validation Accuracy": [accuracy_score(y_val, y_pred_log_reg_val),accuracy_score(y_val, y_pred_svm_val),  accuracy_score(y_val, y_pred_nb_val)],
#     "Test Accuracy": [accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_nb)]
# }

# results = pd.DataFrame(results)
# print(results)

# Initialize the stemmer
stemmer = PorterStemmer()

# Apply stemming and create a new column with the stemmed text
df['text_stemmed'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Recreate the feature matrix with the stemmed text
X_stemmed = vectorizer.fit_transform(df['text_stemmed'])

# Split the dataset into training and testing sets again
X_train_stemmed, X_test_stemmed, y_train, y_test = train_test_split(X_stemmed, df['rating'], test_size=0.2, random_state=42)
X_test_stemmed, X_val_stemmed, y_test, y_val = train_test_split(X_test_stemmed, y_test, test_size=0.5, random_state=42)

# Check the new shape
# print(f"Training set size: {X_train_stemmed.shape}")
# print(f"Testing set size: {X_test_stemmed.shape}")
# print(f"Validation set size: {X_val_stemmed.shape}")

# Define the models
logreg_model = LogisticRegression(max_iter=1000)
svm_model = SVC(kernel='linear')
nb_model = MultinomialNB()


# Logistic Regression
logreg_model.fit(X_train_stemmed, y_train)
y_pred_logreg_stemmed = logreg_model.predict(X_test_stemmed)
y_pred_logreg_val_stemmed = logreg_model.predict(X_val_stemmed)
logreg_accuracy_stemmed = accuracy_score(y_test, y_pred_logreg_stemmed)
logreg_accuracy_val_stemmed = accuracy_score(y_val, y_pred_logreg_val_stemmed)
# print(f"Logistic Regression Accuracy (Stemmed): {logreg_accuracy_stemmed}")
# print(f"Logistic Regression Validation Accuracy (Stemmed): {logreg_accuracy_val_stemmed}")
# print(f"Logistic Regression Training Accuracy (Stemmed): {accuracy_score(y_train, logreg_model.predict(X_train_stemmed))}")


# Support Vector Machine (SVM)
svm_model.fit(X_train_stemmed, y_train)
y_pred_svm_stemmed = svm_model.predict(X_test_stemmed)
y_pred_svm_val_stemmed = svm_model.predict(X_val_stemmed)

svm_acc_stemmed = accuracy_score(y_test, y_pred_svm_stemmed)
svm_acc_val_stemmed = accuracy_score(y_val, y_pred_svm_val_stemmed)
# print(f"SVM Accuracy (Stemmed): {svm_acc_stemmed}")
# print(f"SVM Validation Accuracy (Stemmed): {svm_acc_val_stemmed}") 
# print(f"SVM Training Accuracy (Stemmed): {accuracy_score(y_train, svm_model.predict(X_train_stemmed))}")

# Naïve Bayes
nb_model.fit(X_train_stemmed, y_train)
y_pred_nb_stemmed = nb_model.predict(X_test_stemmed)
nb_acc_stemmed = accuracy_score(y_test, y_pred_nb_stemmed)
# print(f"Naïve Bayes Accuracy (Stemmed): {nb_acc_stemmed:.4f}")

# Convert the text data to a numeric format using CountVectorizer
# vectorizer = CountVectorizer(stop_words='english')
# X = vectorizer.fit_transform(df['text'])

# Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, df['rating'], test_size=0.2, random_state=42)


# Feature Extraction using CountVectorizer with additional settings
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['text'])

# Split the data into training and testing sets
X_train_ngram, X_test_ngram, y_train, y_test = train_test_split(X, df['rating'], test_size=0.2, random_state=42)
X_test_ngram, X_val_ngram, y_test, y_val = train_test_split(X_test_ngram, y_test, test_size=0.5, random_state=42)

# print(f"Shape of feature matrix after using n-grams: {X.shape}")

# Initialize classifiers
classifiers = {
    # "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter to avoid convergence issues
     "SVM (Support Vector Machine)": SVC(),
    # "Naive Bayes": MultinomialNB()
}

# Loop through each classifier, train, and evaluate
for clf_name, clf in classifiers.items():
    # Train the classifier on the training data
    clf.fit(X_train_ngram, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test_ngram)

    y_pred_val = clf.predict(X_val_ngram)

    y_pred_train = clf.predict(X_train_ngram)

    # Evaluate the accuracy
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    # print(f"Accuracy with {clf_name} using n-grams: {accuracy_test:.4f}")
    # print(f"Validation Accuracy with {clf_name} using n-grams: {accuracy_val:.4f}")
    # print(f"Training Accuracy with {clf_name} using n-grams: {accuracy_score(y_train, y_pred_train):.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_test_split(X, y, test_size=0.2, random_state=42)

logreg_model = LogisticRegression(max_iter=1000)

# Logistic Regression with class_weight='balanced'
logreg_model = LogisticRegression(class_weight='balanced')

# SVM with class_weight='balanced'
svm_model = SVC(class_weight='balanced')


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models with relevant parameters
logreg_model = LogisticRegression(max_iter=1000, class_weight='balanced')
svm_model = SVC(class_weight='balanced')
nb_model = MultinomialNB()

# Train the models
logreg_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

# Evaluate the models
logreg_accuracy = logreg_model.score(X_test, y_test)
svm_accuracy = svm_model.score(X_test, y_test)
nb_accuracy = nb_model.score(X_test, y_test)

# Print accuracy
# print(f"Logistic Regression Accuracy: {logreg_accuracy}")
# print(f"SVM Accuracy: {svm_accuracy}")
# print(f"Naïve Bayes Accuracy: {nb_accuracy}")

# Logistic Regression hyperparameters grid
param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [500, 1000],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
# print("Best parameters for Logistic Regression Using Grid Search:", grid_search.best_params_)

# Using best parameters for Logistic Regression
logreg_best = LogisticRegression(C=0.1, max_iter=500, class_weight=None)
logreg_best.fit(X_train, y_train)

# Evaluate on test set
logreg_best_accuracy = logreg_best.score(X_test, y_test)
# print(f"Logistic Regression Accuracy with Best Parameters Using Grid Search: {logreg_best_accuracy}")

# 1. SVM Model Grid Search
svm_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear'],
    'class_weight': [None, 'balanced']
}

svm_grid_search = GridSearchCV(SVC(), svm_params, cv=5, n_jobs=-1, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)

# print(f"Best parameters for SVM: {svm_grid_search.best_params_}")
# print(f"SVM Best Accuracy: {svm_grid_search.best_score_}")

# 2. Naive Bayes Model Grid Search
nb_params = {
    'alpha': [0.01, 0.1, 0.5, 1, 2, 5]
}

nb_grid_search = GridSearchCV(MultinomialNB(), nb_params, cv=5, n_jobs=-1, scoring='accuracy')
nb_grid_search.fit(X_train, y_train)

# print(f"Best parameters for Naive Bayes: {nb_grid_search.best_params_}")
# print(f"Naive Bayes Best Accuracy: {nb_grid_search.best_score_}")

# 1. Retrain SVM with Best Parameters
svm_best_model = SVC(C=0.1, kernel='linear', class_weight=None)
svm_best_model.fit(X_train, y_train)

svm_test_accuracy = svm_best_model.score(X_test, y_test)
# print(f"SVM Test Accuracy with Best Parameters: {svm_test_accuracy:.4f}")

# 2. Retrain Naive Bayes with Best Parameters
nb_best_model = MultinomialNB(alpha=2)
nb_best_model.fit(X_train, y_train)

nb_test_accuracy = nb_best_model.score(X_test, y_test)
# print(f"Naive Bayes Test Accuracy with Best Parameters: {nb_test_accuracy:.4f}")

# Evaluate Logistic Regression model
logreg_pred = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_precision = precision_score(y_test, logreg_pred, average='weighted')
logreg_recall = recall_score(y_test, logreg_pred, average='weighted')
logreg_f1 = f1_score(y_test, logreg_pred, average='weighted')

# Confusion Matrix for Logistic Regression
logreg_cm = confusion_matrix(y_test, logreg_pred)
logreg_disp = ConfusionMatrixDisplay(confusion_matrix=logreg_cm, display_labels=logreg_model.classes_)
logreg_disp.plot()
# plt.title("Logistic Regression Confusion Matrix")
# plt.show()

# print("Logistic Regression Evaluation:")
# print(f"Accuracy: {logreg_accuracy}")
# print(f"Precision: {logreg_precision}")
# print(f"Recall: {logreg_recall}")
# print(f"F1-Score: {logreg_f1}")
# print("\n")

# Evaluate SVM model
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average='weighted')
svm_recall = recall_score(y_test, svm_pred, average='weighted')
svm_f1 = f1_score(y_test, svm_pred, average='weighted')

# Confusion Matrix for SVM
svm_cm = confusion_matrix(y_test, svm_pred)
svm_disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=svm_model.classes_)
svm_disp.plot()
# plt.title("SVM Confusion Matrix")
# plt.show()

# print("SVM Evaluation:")
# print(f"Accuracy: {svm_accuracy}")
# print(f"Precision: {svm_precision}")
# print(f"Recall: {svm_recall}")
# print(f"F1-Score: {svm_f1}")
# print("\n")

# Evaluate Naive Bayes model
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred, average='weighted')
nb_recall = recall_score(y_test, nb_pred, average='weighted')
nb_f1 = f1_score(y_test, nb_pred, average='weighted')

# Confusion Matrix for Naive Bayes
nb_cm = confusion_matrix(y_test, nb_pred)
nb_disp = ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=nb_model.classes_)
nb_disp.plot()
# plt.title("Naive Bayes Confusion Matrix")
# plt.show()

# print("Naive Bayes Evaluation:")
# print(f"Accuracy: {nb_accuracy}")
# print(f"Precision: {nb_precision}")
# print(f"Recall: {nb_recall}")
# print(f"F1-Score: {nb_f1}")
