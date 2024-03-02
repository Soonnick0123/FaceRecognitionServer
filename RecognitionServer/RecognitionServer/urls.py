from django.contrib import admin
from django.urls import path, include
from RecognitionServer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello_world', views.hello_world),
    path('second_message', views.second_function),
]