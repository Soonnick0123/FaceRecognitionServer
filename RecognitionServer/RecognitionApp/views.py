from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,JsonResponse
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from .models import *
from . import serializers
import os
import uuid
import base64

@csrf_exempt
def helloWorld ( request ):
    return JsonResponse({ 'message': 'Hello, world!' })

@csrf_exempt
def secondFunction(request):
    return JsonResponse({'message': "Second Message!"})

@csrf_exempt
def registerCustomer(request):
    name = request.POST['name']
    email = request.POST["email"]
    phone = request.POST["phone"]
    gender = request.POST["gender"]
    # photo = request.FILES.get("photo") # open if use file input
    photo = request.POST.get('photo')

    check_exits=Customer.objects.filter(email=email)

    phone_validator = RegexValidator(regex=r'^\+?1?\d{9,15}$',
                                     message="Phone number must be entered in the format: '+60123456789'. Up to 15 digits allowed.")

    if check_exits:
        return HttpResponse(status=460)
    if not "@" in email:
        return HttpResponse(status=490)
    try:
        phone_validator(phone)

    except ValidationError as e:
        return JsonResponse({'error': e.message}, status=420)

    # original_filename = photo.name
    # file_extension = original_filename.split('.')[-1]
    # unique_filename = f"{uuid.uuid4()}.{file_extension}"
    format, imgstr = photo.split(';base64,')
    ext = format.split('/')[-1]
    unique_filename = f"{uuid.uuid4()}.{ext}"
    data = ContentFile(base64.b64decode(imgstr), name=unique_filename)

    newCustomer = Customer(name=name,email=email,phone=phone,gender=gender,photo=photo)
    newCustomer.photo.save(unique_filename, data, save=False)
    newCustomer.save()

    return HttpResponse(status=200)

@csrf_exempt
def getCustomerList(request):
    allCustomer = Customer.objects.all()
    serializer = serializers.CustomerSerializer(allCustomer,context={'request': request}, many=True)
    return JsonResponse({'customerList': serializer.data}, status=200, safe=False)

@csrf_exempt
def deleteCustomer(request):
    customerId = request.POST['customerId']

    try:
        customer = Customer.objects.get(id=customerId)
        if customer.photo:
            photo_path = customer.photo.path
            if os.path.isfile(photo_path):
                os.remove(photo_path)
        customer.delete()
        return HttpResponse(status=200)
    except Customer.DoesNotExist:
        return HttpResponse(status=460)
    except Exception as e:
        return JsonResponse({'error': e.message}, status=500)