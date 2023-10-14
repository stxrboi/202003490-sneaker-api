from django.urls import path, include
from . import views

urlpatterns = [
    ##path("",views.home),
    path("",views.sentiment_classifier, name='classify')
]
