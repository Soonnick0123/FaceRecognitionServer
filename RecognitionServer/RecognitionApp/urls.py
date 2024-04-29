from django.urls import path, include
from RecognitionApp import views

urlpatterns = [
    path("helloWorld", views.helloWorld),
    path("secondFunction", views.secondFunction),
    path("registerCustomer", views.registerCustomer),
    path("getCustomerList", views.getCustomerList),
    path("deleteCustomer", views.deleteCustomer),
    path("webcamRecognition", views.webcamRecognition),
    path("getLoginRecord", views.getLoginRecord),
    path("deleteRecord", views.deleteRecord),
    path("capturePhotos", views.capturePhotos),
]