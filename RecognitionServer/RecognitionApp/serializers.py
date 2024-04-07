from rest_framework import serializers
from .models import *

class CustomerSerializer(serializers.ModelSerializer):
    photo_url = serializers.SerializerMethodField()
    class Meta:
        model = Customer
        fields = ['id', 'name', 'email', 'phone', 'gender', 'photo', 'photo_url']

    def get_photo_url(self, obj):
        if obj.photo:
            return obj.photo.url
        return None
