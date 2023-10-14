import pickle
import string
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import nltk
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('sneakers_Reviews_Dataset.csv', sep=";")
print('Shape:', df.shape)



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


# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=20)

# Fit and transform the cleaned review_text
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_review_text'])

# Convert the TF-IDF matrix to a Pandas DataFrame
#numeric_review = pd.DataFrame(tfidf_matrix.toarray(
#), columns=tfidf_vectorizer.get_feature_names_out())

# Define sentiment threshold values
positive_threshold = 4.0  # threshold for positive sentiment
neutral_threshold_low = 2.0  # lower threshold for neutral sentiment
neutral_threshold_high = 4.0  # upper threshold for neutral sentiment

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
#print(df[['rating', 'sentiment']])

# Define the features (X) and the target (y)
tfidf_vectorizer = TfidfVectorizer(max_features=20)
X = tfidf_vectorizer.fit_transform(df['cleaned_review_text'])
y = df['sentiment']

# Split the dataset into training and testing sets (e.g., 80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# from sklearn.metrics import classification_report

# Initialize and train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)


# Evaluate the model's performance
report = classification_report(y_test, y_pred)

# Save Model and transformer vectorizer
with open('sentimentlabel.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)


with open('transformvec.pkl', 'wb') as transformvec_file:
    pickle.dump(tfidf_vectorizer, transformvec_file)

print("It is done")