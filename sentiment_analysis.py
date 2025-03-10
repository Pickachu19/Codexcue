# Import necessary libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import random

# Download NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Preprocess the text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(' '.join(text))
    tokens = [word.lower() for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

processed_documents = [(preprocess_text(doc), category) for doc, category in documents]

# Prepare the data for the model
texts = [' '.join(doc) for doc, category in processed_documents]
labels = [1 if category == 'pos' else 0 for doc, category in processed_documents]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

max_length = 100
data = pad_sequences(sequences, maxlen=max_length)
labels = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build and train the model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Debugging information
print(f'Number of samples: {len(labels)}')
print(f'Number of unique tokens: {len(word_index)}')
print(f'Shape of data tensor: {data.shape}')
print(f'Shape of label tensor: {labels.shape}')
print(f'Shape of training data: {X_train.shape}')
print(f'Shape of test data: {X_test.shape}')
print(f'Shape of training labels: {y_train.shape}')
print(f'Shape of test labels: {y_test.shape}')
