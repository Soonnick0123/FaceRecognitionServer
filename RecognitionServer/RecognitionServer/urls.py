from django.contrib import admin
from django.urls import path, include
from RecognitionServer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("helloWorld", views.helloWorld),
    path("secondWessage", views.secondFunction),
]