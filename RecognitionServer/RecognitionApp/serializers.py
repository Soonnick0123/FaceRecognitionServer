from rest_framework import serializers
from .models import *

class CustomerSerializer(serializers.ModelSerializer):
    photo_url = serializers.SerializerMethodField()
    gender = serializers.SerializerMethodField()
    class Meta:
        model = Customer
        fields = ['id', 'name', 'email', 'phone', 'gender', 'photo', 'photo_url']

    def get_gender(self, obj):
        return obj.get_gender_display()

    def get_photo_url(self, obj):
        if obj.photo:
            return self.context['request'].build_absolute_uri(obj.photo.url)
        return None
