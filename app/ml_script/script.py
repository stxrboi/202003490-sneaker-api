import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import pickle

df = pd.read_csv('sneakers_Reviews_Dataset.csv', sep=";")
print('Shape:', df.shape)
df.head()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Apply the preprocess_text function to the 'review_text' column
df['cleaned_review_text'] = df['review_text'].apply(preprocess_text)

print(df.head())

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  

# Fit and transform the cleaned review_text
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_review_text'])

# Convert the TF-IDF matrix to a Pandas DataFrame
numeric_review = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Define sentiment threshold values
positive_threshold = 4.0  # Example threshold for positive sentiment
neutral_threshold_low = 2.0  # Example lower threshold for neutral sentiment
neutral_threshold_high = 4.0  # Example upper threshold for neutral sentiment

# Define a mapping between ratings and sentiment labels
def classify_sentiment(predicted_rating):
    if predicted_rating >= positive_threshold:
        return "Positive"
    elif predicted_rating >= neutral_threshold_low and predicted_rating < neutral_threshold_high:
        return "Neutral"
    else:
        return "Negative"

# Apply the mapping function to create a new 'sentiment' column
df['sentiment'] = df['rating'].apply(classify_sentiment)

# Display the resulting DataFrame with sentiment labels
print(df[['rating', 'sentiment']])

# Define the features (X) and the target (y)
tfidf_vectorizer = TfidfVectorizer(max_features=20) 
X = tfidf_vectorizer.fit_transform(df['cleaned_review_text'])
y = df['sentiment']

# Split the dataset into training and testing sets (e.g., 80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import classification_report

# Initialize and train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)
#y_pred2 = classifier.predict(X_test)

# Evaluate the model's performance
report = classification_report(y_test, y_pred)

# Initialize and train a SVM classifier
classifier2 = SVC(kernel='poly', C=1.0)  
classifier2.fit(X_train, y_train)

# Make predictions on the test data
y_pred3 = classifier2.predict(X_test)

# Evaluate the model's performance
report2 = classification_report(y_test, y_pred3)
print(report)
print(report2)

with open('sentimentlabel.pkl','wb') as model_file:
    pickle.dump(classifier, model_file)

with open('sentimentlabel2.pkl','wb') as model_file2:
    pickle.dump(classifier2, model_file2)

with open('transformvec.pkl','wb') as transformvec_file:
    pickle.dump(tfidf_vectorizer, transformvec_file)