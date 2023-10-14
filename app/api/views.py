from django.shortcuts import render #, redirect
from django.http import JsonResponse
#from rest_framework.response import Response
#from rest_framework.decorators import api_view
#from rest_framework import status
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pickle
import string
#import json


def sentiment_classifier(request):
    if request.method == "POST":
        with open('ml_script/sentimentlabel.pkl','rb') as model_file:
            model = pickle.load(model_file)
        with open('ml_script/transformvec.pkl','rb') as transformvec_file:
            tfidf_vectorizer = pickle.load(transformvec_file)
        review = request.POST.get('review','')
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

        data_vec = tfidf_vectorizer.transform([review])
        classific = model.predict(data_vec)
        return JsonResponse({'sentiment': classific[0]})
        #return Response({'sentiment': classific[0]})
    return render(request, 'home.html')
