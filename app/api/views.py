from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.sentiment import SentimentIntensityAnalyzer


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import string
import json




"""
# Create your views here.
def home(request):
    data = {
        "it works"
    }
    return JsonResponse(data)

@api_view(['GET', 'POST'])
"""
def sentiment_classifier(request):
    """
    if request.method == "GET":
        return Response({'message': 'Sentiment Classifier !'}, status=status.HTTP_200_OK)
    """
    if request.method == "POST":
        with open('ml_script/sentimentlabel.pkl','rb') as model_file:
            model = pickle.load(model_file)
        with open('ml_script/transformvec.pkl','rb') as transformvec_file:
            tfidf_vectorizer = pickle.load(transformvec_file)
        #data = json.loads(request.body)
        #data = data.get('review', '')
        review = request.POST.get('review','')
        #review = ['review']
        #for i in range(len(review)):
        # Tokenize text
        tokens = word_tokenize(review)

            #convert all words to lowercase
        tokens = [word.lower() for word in tokens]
            
            # Remove punctuation
        tokens = [word for word in tokens if word not in string.punctuation]

            # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # Join tokens back into a string
        review = ' '.join(tokens)
       # tfidf_vectorizer = TfidfVectorizer(max_features=20) 
        data_vec = tfidf_vectorizer.transform([review])

        classific = model.predict(data_vec)

        return JsonResponse({'sentiment': classific[0]})
        #return Response({'sentiment': classific[0]})
    return render(request, 'home.html')
            


#@api_view(['POST'])
#
        