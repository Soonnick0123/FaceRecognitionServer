from rest_framework import serializers
from .models import *

class CustomerSerializer(serializers.ModelSerializer):
    photo1_url = serializers.SerializerMethodField()
    photo2_url = serializers.SerializerMethodField()
    photo3_url = serializers.SerializerMethodField()
    gender = serializers.SerializerMethodField()
    class Meta:
        model = Customer
        fields = ['id', 'name', 'username', 'email', 'phone', 'gender', 'photo1_url', 'photo2_url', 'photo3_url']

    def get_gender(self, obj):
        return obj.get_gender_display()

    def get_photo1_url(self, obj):
        if obj.photo1:
            return self.context['request'].build_absolute_uri(obj.photo1.url)
        return None

    def get_photo2_url(self, obj):
        if obj.photo2:
            return self.context['request'].build_absolute_uri(obj.photo2.url)
        return None

    def get_photo3_url(self, obj):
        if obj.photo3:
            return self.context['request'].build_absolute_uri(obj.photo3.url)
        return None

class LoginRecordSerializer(serializers.ModelSerializer):
    customer = CustomerSerializer(read_only=True)  # Using the existing CustomerSerializer

    class Meta:
        model = LoginRecord
        fields = ['id','customer', 'login_time']