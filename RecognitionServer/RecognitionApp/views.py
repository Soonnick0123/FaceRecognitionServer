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

import insightface
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import time

from anti_spoof.test import test
from anti_spoof.src.anti_spoof_predict import AntiSpoofPredict

embedding_file_path = 'central_embeddings.npy'
# model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider']) # default is buffalo_l
model_insight = insightface.app.FaceAnalysis() # default is buffalo_l
model_insight.prepare(ctx_id=1) # 0 -- GPU , 1 -- CPU
model_antispoof = AntiSpoofPredict(0,'C:\\Users\\yanha\\Desktop\\test\\Silent_Face_Anti_Spoofing\\resources\\anti_spoof_models')



@csrf_exempt
def helloWorld ( request ):
    return JsonResponse({ 'message': 'Hello, world!' })

@csrf_exempt
def secondFunction(request):
    return JsonResponse({'message': "Second Message!"})

# @csrf_exempt
# def registerCustomer(request):
#     username = request.POST['username']
#     name = request.POST['name']
#     email = request.POST["email"]
#     phone = request.POST["phone"]
#     gender = request.POST["gender"]
#     # photo = request.FILES.get("photo") # open if use file input
#     photos = [request.POST.get(f'photo{i}') for i in range(1, 4)]


#     check_exits=Customer.objects.filter(username=username)

#     phone_validator = RegexValidator(regex=r'^\+?1?\d{9,15}$',
#                                      message="Phone number must be entered in the format: '+60123456789'. Up to 15 digits allowed.")

#     username_validator = RegexValidator(
#         regex=r'^[a-zA-Z0-9]*$',
#         message="Username can only contain letters and numbers.")

#     if check_exits:
#         return HttpResponse(status=460)
#     if not "@" in email:
#         return HttpResponse(status=490)
#     try:
#         phone_validator(phone)
#         username_validator(username)

#     except ValidationError as e:
#         return JsonResponse({'error': e.message}, status=420)

#     # original_filename = photo.name
#     # file_extension = original_filename.split('.')[-1]
#     # unique_filename = f"{uuid.uuid4()}.{file_extension}"

#     # format, imgstr = photo.split(';base64,')                            # close if use file input
#     # ext = format.split('/')[-1]                                         # close if use file input
#     # unique_filename = f"{uuid.uuid4()}.{ext}"                           # close if use file input
#     # data = ContentFile(base64.b64decode(imgstr), name=unique_filename)  # close if use file input

#     newCustomer = Customer(name=name,username=username,email=email,phone=phone,gender=gender)
#     for i, photo_data in enumerate(photos, start=1):
#         if photo_data:
#             format, imgstr = photo_data.split(';base64,')
#             ext = format.split('/')[-1]
#             unique_filename = f"{uuid.uuid4()}.{ext}"
#             data = ContentFile(base64.b64decode(imgstr), name=unique_filename)
#             getattr(newCustomer, f'photo{i}').save(unique_filename, data, save=False)
#     newCustomer.save()

#     return HttpResponse(status=200)

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

    embedding_file_path = 'central_embeddings.npy'

#YH CODE

def expand_bbox(frame, bbox, expand_margin=0.1):
    """
    :param expand_margin: margin to expand
    :return: expanded bbox
    """
    h, w = frame.shape[:2]  # 获取图像的高度和宽度
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # 计算扩大的大小
    expand_width = width * expand_margin
    expand_height = height * expand_margin

    # 扩大边界框，同时确保不超出图像边界
    x_min_expanded = max(0, int(x_min - expand_width))
    y_min_expanded = max(0, int(y_min - expand_height))
    x_max_expanded = min(w, int(x_max + expand_width))
    y_max_expanded = min(h, int(y_max + expand_height))

    return [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded]

def adjust_to_aspect_ratio(frame, bbox, target_ratio=3/4):
    """
    adjust the image to aspect ratio
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        width_diff = width - new_width
        x_min += width_diff // 2
        x_max -= width_diff // 2
    elif current_ratio < target_ratio:
        new_height = int(width / target_ratio)
        height_diff = height - new_height
        y_min += height_diff // 2
        y_max -= height_diff // 2

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

    return frame[y_min:y_max, x_min:x_max]

def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embedding_to_string(embedding):
    """Convert an embedding to a string representation with limited precision."""
    return np.array2string(embedding, precision=6, separator=',', max_line_width=np.inf)[1:-1]
    # return str(embedding)

def string_to_embedding(embedding_str):
    """Convert a string representation back to a numpy array."""
    return np.fromstring(embedding_str, dtype=float, sep=',')
    # return np.array(embedding_str)

def find_similar_embedding(current_embedding, tracked_embedding, similarity_threshold=0.85):
    """Find a key for a similar embedding."""
    for key in tracked_embedding.keys():
        stored_embedding = string_to_embedding(key)
        # print(cosine_similarity(current_embedding, stored_embedding))
        if cosine_similarity(current_embedding, stored_embedding) > similarity_threshold:
            return key
    # If no similar embedding is found, convert the current embedding to a string and use it as a new key
    return embedding_to_string(current_embedding)

def load_embeddings(file_path=embedding_file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except (FileNotFoundError, EOFError):
        print(f"Warning: {file_path} not found or is empty. Starting with empty data.")
        data = {}  # Start with an empty dictionary
    return data

def find_closest_match(query_embedding, data):
    min_distance = float('inf')
    matched_label = "Unknown"
    for label, person_embeddings in data.items():
        for emb in person_embeddings:
            distance = cosine(query_embedding, emb)
            if distance < min_distance:
                min_distance = distance
                matched_label = label
    return min_distance, matched_label

def save_embeddings(data, file_path):
    """Save embeddings and labels to a file."""
    np.save(file_path, data)

def is_unique_embedding(new_embedding, existing_embeddings, threshold=0.05):
    """
    Check if the new_embedding is sufficiently different from all existing_embeddings.
    A very small threshold is used to only exclude nearly identical embeddings.
    """
    for embedding in existing_embeddings:
        # print(cosine(new_embedding, embedding))
        if cosine(new_embedding, embedding) < threshold or cosine(new_embedding, embedding) > 0.85 :
            # The new embedding is nearly identical to an existing one
            if cosine(new_embedding, embedding) < threshold:
                print("Embedding Data Existed")
            elif cosine(new_embedding, embedding) > 0.85 :
                print("Different People/Face")
            return False

    return True

def find_largest_bbox(faces):
    max_area = 0
    largest_face = None

    if faces is not None:
        for face in faces:
            x_min, y_min, x_max, y_max = face.bbox
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                largest_face = face

    return largest_face

def register_dataset(dataset_path, model, embeddings_file):
    """register all customer using a dataset folder"""
    data = load_embeddings(embeddings_file)
    # read every sub folder
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue  # skip not folder type

        print(f"Processing {person_name}...")
        if person_name not in data:
            data[person_name] = []  # Initialize the list for new individuals

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            faces = model.get(img)
            face = find_largest_bbox(faces)
            if face:
                embedding = face.embedding
                if is_unique_embedding(embedding, data[person_name]):
                    data[person_name].append(embedding)
                    print(f"{person_name}, updated, {img_name}")
                else:
                    print(f"Duplicate embedding detected for {person_name}, skipped.")
            else:
                print(f"Skipped {img_name}: 0 face detected.")

    save_embeddings(data, embeddings_file)

def register_person_face(dataset_path, username, model, embeddings_file):
    data = load_embeddings(embeddings_file)
    if username not in data:
        data[username] = []  # Initialize the list for new individuals
    else:
        return print("User existed")
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        faces = model.get(img)
        if len(faces) > 1:
            return print("More than 1 face detected")
        for face in faces:
            if face:
                embedding = face.embedding
                if is_unique_embedding(embedding, data[username]):
                    data[username].append(embedding)
                    print(f"{username}, updated, {img_name}")
                else:
                    print(f"Duplicate embedding / different people detected for {username}, skipped.")
            else:
                print(f"Skipped {img_name}: 0 face detected.")

    save_embeddings(data, embeddings_file)

@csrf_exempt
def yh_recognition(request):
    cap = cv2.VideoCapture(0)
    tracked_embedding={}

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        faces = model_insight.get(frame)
        current_time = time.time()
        for face in faces:
            bbox = face.bbox.astype(int)
            face_id = face.embedding

            expanded_bbox = expand_bbox(frame, bbox, expand_margin=0.5)
            face_img_adjusted = adjust_to_aspect_ratio(frame, expanded_bbox, target_ratio=3/4)
            face_img = face_img_adjusted

            # 进行欺骗检测
            is_real = test(
                        image=face_img,
                        model_dir='C:\\Users\\yanha\\Desktop\\test\\Silent_Face_Anti_Spoofing\\resources\\anti_spoof_models',
                        model_test =model_antispoof

            )

            embedding_key = find_similar_embedding(face_id, tracked_embedding, 0.8)
            if embedding_key not in tracked_embedding:
                tracked_embedding[embedding_key] = {'spoofCount': 0, 'realCount':0 , 'approved':False ,'first_detected': current_time}

            if is_real == 1:
                # If the face is detected as real in previous frames
                if tracked_embedding[embedding_key]['approved'] is True:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    query_embedding = face.embedding
                    min_distance, matched_label = find_closest_match(query_embedding, load_embeddings(embedding_file_path))
                    if min_distance < 0.5:
                        cv2.putText(frame, matched_label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                # Overall Detect as Real
                elif (current_time - tracked_embedding[embedding_key]['first_detected'] >= 2.5) and ((current_time - tracked_embedding[embedding_key]['first_detected'] <= 3)) and (tracked_embedding[embedding_key]['spoofCount'] < 10) and (tracked_embedding[embedding_key]['realCount'] > 10):
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    query_embedding = face.embedding
                    min_distance, matched_label = find_closest_match(query_embedding, load_embeddings(embedding_file_path))
                    if min_distance < 0.5:
                        cv2.putText(frame, matched_label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        tracked_embedding[embedding_key]['approved'] = True

                # Scanning and Detect as Real
                elif (current_time - tracked_embedding[embedding_key]['first_detected'] < 3) and (tracked_embedding[embedding_key]['spoofCount'] < 10):
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (135,206,235), 2)
                    tracked_embedding[embedding_key]['realCount'] += 1

                # Scanning and Detect as Spoof
                elif (current_time - tracked_embedding[embedding_key]['first_detected'] < 3) and (tracked_embedding[embedding_key]['spoofCount'] >= 10):
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, "Spoof1", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                tracked_embedding[embedding_key]['spoofCount'] += 1
                if tracked_embedding[embedding_key]['approved'] is True:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    query_embedding = face.embedding
                    min_distance, matched_label = find_closest_match(query_embedding, load_embeddings(embedding_file_path))
                    if min_distance < 0.5:
                        cv2.putText(frame, matched_label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                elif (current_time - tracked_embedding[embedding_key]['first_detected'] < 3) and (tracked_embedding[embedding_key]['spoofCount'] < 5):
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (135,206,235), 2)
                else:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    text = "Count: " + str(tracked_embedding[embedding_key]['spoofCount'])
                    cv2.putText(frame,text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)


            # tracked_embedding[embedding_key]['spoofCount'] += 1
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            # text = "Count: " + str(tracked_embedding[embedding_key]['spoofCount'])
            # cv2.putText(frame,text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        for key in list(tracked_embedding.keys()):
            if (current_time - tracked_embedding[key]['first_detected'] > 3) and (tracked_embedding[key]['approved'] is False):
                del tracked_embedding[key]
            elif (current_time - tracked_embedding[key]['first_detected'] > 10) and (tracked_embedding[key]['approved'] is True):
                del tracked_embedding[key]

        cv2.imshow('Live Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@csrf_exempt
def registerCustomer(request):
    username = request.POST['username']
    name = request.POST['name']
    email = request.POST["email"]
    phone = request.POST["phone"]
    gender = request.POST["gender"]
    # photo = request.FILES.get("photo") # open if use file input
    photos = [request.POST.get(f'photo{i}') for i in range(1, 4)]

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

    for photo in photos:
        format, imgstr = photo.split(';base64,')
        ext = format.split('/')[-1]
        unique_filename = f"{uuid.uuid4()}.{ext}"
        data = ContentFile(base64.b64decode(imgstr), name=unique_filename)

        image_array = np.frombuffer(data.read(), np.uint8)  # Read the content as a byte array
        photo = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Decode byte array as a color image


        faces = model_insight.get(photo)
        if len(faces) > 1:
            print("more than 1 detected")
            return JsonResponse({'error': "More than 1 faces detected"}, status=501)

        elif len(faces) == 1:
            for face in faces:
                is_real = test(
                            image=photo,
                            model_dir='C:\\Users\\yanha\\Desktop\\test\\Silent_Face_Anti_Spoofing\\resources\\anti_spoof_models',
                            model_test = model_antispoof
                )

                if is_real != 1:
                    return JsonResponse({'error': "Spoof Detected"}, status=502)

    newCustomer = Customer(name=name,username=username,email=email,phone=phone,gender=gender)
    for i, photo_data in enumerate(photos, start=1):
        if photo_data:
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]
            unique_filename = f"{uuid.uuid4()}.{ext}"
            data = ContentFile(base64.b64decode(imgstr), name=unique_filename)
            getattr(newCustomer, f'photo{i}').save(unique_filename, data, save=False)
    newCustomer.save()

    # update face recognition model
    relative_path = f'./media/customer_photos/{username}'
    customer_path = os.path.abspath(relative_path)
    if os.path.exists(customer_path):
        register_person_face(customer_path, username, model_insight, embedding_file_path)
    else:
        print("path not exist")
        return JsonResponse({'error': "Path Error"}, status=503)

    return HttpResponse(status=200)