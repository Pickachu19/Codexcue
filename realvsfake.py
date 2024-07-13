# real_vs_fake_news_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import re
import os

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
# Preprocess data
def preprocess_data(df):
    df.dropna(inplace=True)
    
    # Convert labels to numeric if necessary
    if df['label'].dtype == object:
        df['label'] = df['label'].apply(lambda x: 1 if x.strip().lower() == 'real' else 0)
    
    df['text'] = df['title'] + " " + df['text']
    df['text'] = df['text'].apply(clean_text)
    return df

# Feature extraction
def extract_features(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label']
    return X, y, vectorizer


# Feature extraction
def extract_features(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label']
    return X, y, vectorizer

# Train-test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Neural Network Model
def train_neural_network(X_train, y_train):
    nn_model = Sequential()
    nn_model.add(Dense(512, input_shape=(5000,), activation='relu'))
    nn_model.add(Dropout(0.5))
    nn_model.add(Dense(256, activation='relu'))
    nn_model.add(Dropout(0.5))
    nn_model.add(Dense(1, activation='sigmoid'))
    
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    return nn_model

# Model evaluation
def evaluate_model(model, X_test, y_test, model_type="Logistic Regression"):
    if model_type == "Logistic Regression":
        y_pred = model.predict(X_test)
    else:
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    # Dataset file path
    dataset_file = 'news_articles.csv'
    
    # Load dataset
    df = load_dataset(dataset_file)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Feature extraction
    X, y, vectorizer = extract_features(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("Evaluating Logistic Regression model...")
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Train Neural Network model (Optional)
    print("\nTraining Neural Network model...")
    nn_model = train_neural_network(X_train, y_train)
    print("Evaluating Neural Network model...")
    evaluate_model(nn_model, X_test, y_test, "Neural Network")

if __name__ == "__main__":
    main()
