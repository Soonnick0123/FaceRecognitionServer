from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,JsonResponse
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from pathlib import Path
from .models import *
from . import serializers
import os
import uuid
import base64
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import pandas as pd
from tempfile import NamedTemporaryFile

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

        customer.delete()
        return HttpResponse(status=200)
    except Customer.DoesNotExist:
        return HttpResponse(status=460)
    except Exception as e:
        return JsonResponse({'error': e.message}, status=500)

@csrf_exempt
def webcamRecognition(request):
    base64photo = request.POST.get('photo')
    format, imgstr = base64photo.split(';base64,')
    img_data = base64.b64decode(imgstr)
    img = Image.open(BytesIO(img_data))

    with NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        img.save(tmp_file, format='PNG')
        tmp_file_path = tmp_file.name

    try:
        customer_photos_path = os.path.join(settings.MEDIA_ROOT, 'customer_photos')
        results = DeepFace.find(img_path=tmp_file_path,
                                db_path=customer_photos_path,
                                model_name="Facenet512",
                                detector_backend="retinaface",
                                enforce_detection=False,
                                threshold=0.3)
        if results:
            first_result = results[0]
            if isinstance(first_result, pd.DataFrame):
                if first_result.empty:
                    return HttpResponse(status=420)
            first_result = first_result.iloc[0]
            identity = first_result.get('identity')
            distance = first_result.get('distance')
            identity_path = Path(identity)
            customer_username = identity_path.parts[-2]
            print(customer_username,distance)
            try:
                customer = Customer.objects.get(username=customer_username)
                record_login(customer)
                return HttpResponse(status=200)
            except Customer.DoesNotExist:
                return HttpResponse(status=420)
        else:
            return HttpResponse(status=420)

    except Exception as e:
        return JsonResponse({'error': e.message}, status=440)

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@csrf_exempt
def record_login(customer):
    LoginRecord.objects.create(customer=customer, login_time=timezone.now())
    return

@csrf_exempt
def deleteRecord(request):
    recordId = request.POST['recordId']
    try:
        record = LoginRecord.objects.get(id=recordId)
        record.delete()
        return HttpResponse(status=200)
    except LoginRecord.DoesNotExist:
        return HttpResponse(status=420)
    except Customer.DoesNotExist:
        return HttpResponse(status=420)

@csrf_exempt
def getLoginRecord(request):
    allRecord = LoginRecord.objects.all().order_by('-login_time')
    serializer = serializers.LoginRecordSerializer(allRecord,context={'request': request}, many=True)
    return JsonResponse({'recordList': serializer.data}, status=200, safe=False)