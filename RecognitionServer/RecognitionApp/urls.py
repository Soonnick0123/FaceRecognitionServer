from django.urls import path, include
from RecognitionApp import views

urlpatterns = [
    path("helloWorld", views.helloWorld),
    path("secondFunction", views.secondFunction),
]