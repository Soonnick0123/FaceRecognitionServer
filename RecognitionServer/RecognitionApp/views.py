from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,JsonResponse
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from deepface import DeepFace
from deepface.commons import package_utils
from deepface.modules import recognition, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger
from pathlib import Path
from .models import *
from . import serializers
import os
import uuid
import base64
import pickle
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
from tempfile import NamedTemporaryFile

customer_photos_path = os.path.join(settings.MEDIA_ROOT, 'customer_photos')
representationModel = []

def loadDeepfaceRepresentationModel():
    detector_backend="retinaface"
    enforce_detection=False
    model_name = "Facenet512"

    logger = Logger(module="deepface/modules/recognition.py")
    file_name = f"ds_{model_name}_{detector_backend}_v2.pkl"
    file_name = file_name.replace("-", "").lower()
    datastore_path = os.path.join(customer_photos_path, file_name)
    model: FacialRecognition = modeling.build_model(model_name)
    target_size = model.input_shape
    align: bool = True
    normalization: str = "base"
    silent: bool = False
    tic = time.time()
    df_cols = [
        "identity",
        "hash",
        "embedding",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    # Ensure the proper pickle file exists
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as f:
            pickle.dump([], f)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    # check each item of representations list has required keys
    for i, current_representation in enumerate(representations):
        missing_keys = list(set(df_cols) - set(current_representation.keys()))
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # embedded images
    pickled_images = [representation["identity"] for representation in representations]

    # Get the list of images on storage
    storage_images = recognition.__list_images(path=customer_photos_path)

    if len(storage_images) == 0:
        raise ValueError(f"No item found in {customer_photos_path}")

    # Enforce data consistency amongst on disk images and pickle file
    must_save_pickle = False
    new_images = list(set(storage_images) - set(pickled_images))  # images added to storage
    old_images = list(set(pickled_images) - set(storage_images))  # images removed from storage

    # detect replaced images
    replaced_images = []
    for current_representation in representations:
        identity = current_representation["identity"]
        if identity in old_images:
            continue
        alpha_hash = current_representation["hash"]
        beta_hash = package_utils.find_hash_of_file(identity)
        if alpha_hash != beta_hash:
            logger.debug(f"Even though {identity} represented before, it's replaced later.")
            replaced_images.append(identity)

    if not silent and (len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0):
        logger.info(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s)."
        )

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images = new_images + replaced_images
    old_images = old_images + replaced_images

    # remove old images first
    if len(old_images) > 0:
        representations = [rep for rep in representations if rep["identity"] not in old_images]
        must_save_pickle = True

    # find representations for new images
    if len(new_images) > 0:
        representations += recognition.__find_bulk_embeddings(
            employees=new_images,
            model_name=model_name,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            silent=silent,
        )  # add new images
        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)
        if not silent:
            logger.info(f"There are now {len(representations)} representations in {file_name}")
            return representations
    # Should we have no representations bailout
    if len(representations) == 0:
        if not silent:
            toc = time.time()
            logger.info(f"find function duration {toc - tic} seconds")
            return []
    return representations

representationModel = loadDeepfaceRepresentationModel()

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
    global representationModel
    representationModel = loadDeepfaceRepresentationModel()
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
        global representationModel
        representationModel = loadDeepfaceRepresentationModel()
        customer.delete()

        return HttpResponse(status=200)
    except Customer.DoesNotExist:
        return HttpResponse(status=460)
    except Exception as e:
        return JsonResponse({'error': e}, status=500)

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
        results = DeepFace.find(img_path=tmp_file_path,
                                db_path=customer_photos_path,
                                model_name="Facenet512",
                                detector_backend="retinaface",
                                enforce_detection=False,
                                threshold=0.2,
                                representations = representationModel)
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
        return JsonResponse({'error': e}, status=440)

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