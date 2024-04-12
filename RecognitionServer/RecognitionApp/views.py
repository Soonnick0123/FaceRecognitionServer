from django.shortcuts import render
from django.conf import settings
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
    username = request.POST['username']
    name = request.POST['name']
    email = request.POST["email"]
    phone = request.POST["phone"]
    gender = request.POST["gender"]
    # photo = request.FILES.get("photo") # open if use file input
    photos = [request.POST.get(f'photo{i}') for i in range(1, 4)]


    check_exits=Customer.objects.filter(username=username)

    phone_validator = RegexValidator(regex=r'^\+?1?\d{9,15}$',
                                     message="Phone number must be entered in the format: '+60123456789'. Up to 15 digits allowed.")

    username_validator = RegexValidator(
        regex=r'^[a-zA-Z0-9]*$',
        message="Username can only contain letters and numbers.")

    if check_exits:
        return HttpResponse(status=460)
    if not "@" in email:
        return HttpResponse(status=490)
    try:
        phone_validator(phone)
        username_validator(username)

    except ValidationError as e:
        return JsonResponse({'error': e.message}, status=420)

    # original_filename = photo.name
    # file_extension = original_filename.split('.')[-1]
    # unique_filename = f"{uuid.uuid4()}.{file_extension}"

    # format, imgstr = photo.split(';base64,')                            # close if use file input
    # ext = format.split('/')[-1]                                         # close if use file input
    # unique_filename = f"{uuid.uuid4()}.{ext}"                           # close if use file input
    # data = ContentFile(base64.b64decode(imgstr), name=unique_filename)  # close if use file input

    newCustomer = Customer(name=name,username=username,email=email,phone=phone,gender=gender)
    for i, photo_data in enumerate(photos, start=1):
        if photo_data:
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]
            unique_filename = f"{uuid.uuid4()}.{ext}"
            data = ContentFile(base64.b64decode(imgstr), name=unique_filename)
            getattr(newCustomer, f'photo{i}').save(unique_filename, data, save=False)
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
        folder_path = os.path.join(settings.MEDIA_ROOT, 'customer_photos', customer.username)

        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(folder_path)

        # if customer.photo:
        #     photo_path = customer.photo.path
        #     if os.path.isfile(photo_path):
        #         os.remove(photo_path)
        customer.delete()
        return HttpResponse(status=200)
    except Customer.DoesNotExist:
        return HttpResponse(status=460)
    except Exception as e:
        return JsonResponse({'error': e.message}, status=500)