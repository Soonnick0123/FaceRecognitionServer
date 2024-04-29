from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,JsonResponse
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from deepface.commons.logger import Logger
from deepface.modules import recognition, modeling
from deepface.modules.recognition import find as DeepFind
from deepface.models.FacialRecognition import FacialRecognition
from pathlib import Path
from .models import *
from . import serializers
import os
import uuid
import base64
import asyncio
from asgiref.sync import async_to_sync, sync_to_async
import pickle
from PIL import Image
from io import BytesIO
from deepface import DeepFace
from deepface.commons import package_utils
from tqdm import tqdm
import pandas as pd
from tempfile import NamedTemporaryFile

import insightface
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import time

from anti_spoof.test import test
from anti_spoof.src.anti_spoof_predict import AntiSpoofPredict

import subprocess
import shlex
from deepface import DeepFace

from concurrent.futures import ThreadPoolExecutor, as_completed

embedding_file_path = 'central_embeddings.npy'
model_insight = insightface.app.FaceAnalysis() # default is buffalo_l
model_insight.prepare(ctx_id=1) # 0 -- GPU , 1 -- CPU
model_antispoof = AntiSpoofPredict(0,os.path.abspath('anti_spoof/resources/anti_spoof_models'))
previous_recognition_process = True
previous_register_process = True

representationModel = []
customer_photos_path = os.path.join(settings.MEDIA_ROOT, 'customer_photos')

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

def set_cpu_affinity(core_ids):
    """ Set CPU affinity for the current process or thread to specific cores """
    os.sched_setaffinity(0, core_ids)

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
        all_embeddings = load_embeddings(embedding_file_path)

        if customer.username in all_embeddings:
            del all_embeddings[customer.username]
            save_embeddings(all_embeddings,embedding_file_path)

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


def process_update_deepface(customer_path, username, model_insight, embedding_file_path):
    # global representationModel
    # representationModel = loadDeepfaceRepresentationModel()
    register_person_face(customer_path, username, model_insight, embedding_file_path)

@csrf_exempt
def registerCustomer(request):
    # task=asyncio.create_task(loadReprentationModel())
    username = request.POST['username']
    name = request.POST['name']
    email = request.POST["email"]
    phone = request.POST["phone"]
    gender = request.POST["gender"]
    photos = [request.POST.get(f'photo{i}') for i in range(1, 4)]

    check_exits=Customer.objects.filter(username=username)

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

    update_deepface_list=[]
    # set_cpu_affinity({0})
    # with ThreadPoolExecutor(max_workers=4) as executor:
    # with ThreadPoolExecutor(max_workers=3, initializer=set_cpu_affinity, initargs=({1, 2, 3},)) as executor:
    try:
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
                                model_dir=os.path.abspath('anti_spoof/resources/anti_spoof_models'),
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

        time_test_start = time.time()

        # update face recognition model
        relative_path = f'./media/customer_photos/{username}'
        customer_path = os.path.abspath(relative_path)
        if os.path.exists(customer_path):
            # update_deepface = executor.submit(process_update_deepface,customer_path, username, model_insight, embedding_file_path)
            # update_deepface_list.append(update_deepface)
            global representationModel
            representationModel = loadDeepfaceRepresentationModel()
            register_person_face(customer_path, username, model_insight, embedding_file_path)
            # await asyncio.sleep(1)
            # while True:
            #     for check in list(update_deepface_list):
            #         if check.done():
            #             time_test_end = time.time()
            #             print("time used:",time_test_end - time_test_start)
            #             return HttpResponse(status=200)
            #         #     # result = check.result()
            #         #     # boolean, matched_label = result
            #         #     # print(matched_label)
            #         #     update_deepface_list.remove(check)
            #         # if len(update_deepface_list):
            #         #     customer = Customer.objects.get(username=matched_label)
            #         #     record_login(customer)
            #         #     process.terminate()
            #         #     cv2.destroyAllWindows()
            #         #     return HttpResponse(status=200)
            # print("here")
            time_test_end = time.time()
            print("time used:",time_test_end - time_test_start)

            return HttpResponse(status=200)
        else:
            print("path not exist")
            return JsonResponse({'error': "Path Error"}, status=503)

    except Exception as e:
        print("message: ",e)



def frame_to_base64(frame):
    # 将图像从BGR转换为JPEG格式的字节流
    retval, buffer = cv2.imencode('.jpg', frame)
    if retval:
        # 将JPEG图像编码为Base64字符串
        jpg_as_text = base64.b64encode(buffer).decode()
        return jpg_as_text
    return None


def process_frame_register(frame, model, photos):
    faces = model.get(frame)
    global previous_register_process
    if len(faces)>0:
        previous_register_process = True
        return True,frame
    else:
        previous_register_process = True
        return False,frame

@csrf_exempt
def capturePhotos(request):
    with ThreadPoolExecutor(max_workers=4) as executor:
        width, height = 640, 480
        command = f"libcamera-vid -t 0 --nopreview --inline --width {width} --height {height} --codec yuv420 -o -"
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
        last_saved_time = time.time()
        countdown_start = time.time()
        countdown_duration = 3
        frame_size = width * height * 3 // 2  # YUV420: Y + UV (size = width * height * 1.5)
        photos = []
        face_check_list = []
        global previous_register_process
        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_TOPMOST,1)
        try:
            while True:
                # 读取一帧的 YUV420 数据
                if len(photos) == 3:
                    print("HEY")
                    process.terminate()
                    cv2.destroyAllWindows()
                    break

                raw_image = process.stdout.read(frame_size)
                if len(raw_image) == frame_size:
                    # 将数据转换为 NumPy 数组
                    yuv = np.frombuffer(raw_image, dtype=np.uint8).reshape((height + height // 2, width))
                    # 将 YUV420 转换为 BGR
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

                    overlay = bgr.copy()
                    output = bgr.copy()
                    alpha = 0.5

                    time_elapsed = time.time() - countdown_start
                    remaining_time = countdown_duration - int(time_elapsed)

                    if remaining_time <0 and previous_register_process:
                        countdown_start = time.time()
                    else:
                        if previous_register_process and remaining_time<3:
                            text = "Capturing Photo" + str(len(photos)+1)
                            cv2.putText(overlay, text,(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,0), 4)
                            cv2.putText(overlay, str(remaining_time + 1),(width//2 - 30,height//2), cv2.FONT_HERSHEY_SIMPLEX, 5,(0,0,0), 10)
                        elif previous_register_process is False:
                            text = "Processing Photo" + str(len(photos)+1)
                            cv2.putText(overlay, text,(width//6,height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5)

                    cv2.addWeighted(overlay,alpha,output,1 - alpha,0,output)

                    if previous_register_process:
                        current_time = time.time()
                        if current_time - last_saved_time >= 4:  # 检查时间间隔是否达到1.5秒
                            last_saved_time = current_time  # 更新上一次保存图片的时间
                            previous_register_process = False
                            face_check = executor.submit(process_frame_register, bgr, model_insight,photos)
                            face_check_list.append(face_check)
                            # faces = model_insight.get(bgr)
                            # if len(faces) > 0:
                            #     photos.append(frame_to_base64(bgr))
                    else:
                        countdown_start = time.time()
                        last_saved_time = time.time()

                    # # 显示图像
                    cv2.imshow("Frame", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        process.terminate()
                        cv2.destroyAllWindows()
                        break

                    for check in list(face_check_list):
                        if check.done():
                            result = check.result()
                            boolean, checked_frame = result
                            face_check_list.remove(check)
                            if boolean:
                                photos.append(frame_to_base64(checked_frame))
                                # process.terminate()
                                # cv2.destroyAllWindows()
                                # return HttpResponse(status=200)


        finally:
            process.terminate()
            cv2.destroyAllWindows()
            return JsonResponse({'photos':photos}, status=200, safe=False)


def process_frame(frame, model_insight, model_antispoof,tracked_embedding):
    try:
        faces = model_insight.get(frame)
        current_time = time.time()
        insight_distance = 0
        deep_distance = 0

        insight_label = ""
        deep_label = ""
        global previous_recognition_process

        faces = model_insight.get(frame)
        if len(faces) > 1:
            print("more than 1 detected")
            previous_recognition_process = True
            return False,"More Than 1 Face Detected"

        elif len(faces) == 1:
            for face in faces:
                is_real = test(
                            image=frame,
                            model_dir=os.path.abspath('anti_spoof/resources/anti_spoof_models'),
                            model_test = model_antispoof
                )

                if is_real != 1:
                    previous_recognition_process = True
                    return False,"Spoof Detected"

                query_embedding = face.embedding
                min_distance, matched_label = find_closest_match(query_embedding, load_embeddings(embedding_file_path))
                print("test insightface",matched_label)
                if min_distance < 0.3:
                    insight_distance = min_distance
                    insight_label = matched_label

                    # return True, matched_label

            if insight_label != "":
                deepResult = DeepFind(img_path=frame,
                                    db_path=customer_photos_path,
                                    model_name="Facenet512",
                                    detector_backend="retinaface",
                                    enforce_detection=False,
                                    threshold=0.3,
                                    representations=representationModel)

                if deepResult:
                    first_result = deepResult[0]
                    if isinstance(first_result, pd.DataFrame):
                        if first_result.empty:
                            previous_recognition_process = True
                            return False, "not a member"
                    first_result = first_result.iloc[0]
                    identity = first_result.get('identity')
                    distance = first_result.get('distance')
                    identity_path = Path(identity)
                    customer_username = identity_path.parts[-2]
                    print("Deepface result: ",customer_username,distance)
                    print("insight result: ",insight_label,insight_distance)
                    deep_label = customer_username
                    deep_distance = distance

                    if insight_label==deep_label and abs(deep_distance-insight_distance) < 0.1:
                        previous_recognition_process = True
                        return True,deep_label

                else:
                    previous_recognition_process = True
                    print("deep cannot detect the face")
                    return False,"not a member"
            else:
                previous_recognition_process = True
                print("insight cannot detect the face")
                return False,"not a member"
        previous_recognition_process = True
        return False,"Nothing"

    except Exception as e:
        previous_recognition_process = True
        return False,"Something Error"

@csrf_exempt
def webcamRecognition(request):
    with ThreadPoolExecutor(max_workers=4) as executor:
        width, height = 640, 480
        command = f"libcamera-vid -t 0 --nopreview --inline --width {width} --height {height} --codec yuv420 -o -"
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
        last_saved_time = time.time()
        countdown_start = time.time()
        countdown_duration = 3
        frame_size = width * height * 3 // 2  # YUV420: Y + UV (size = width * height * 1.5)
        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_TOPMOST,1)
        face_check_list = []
        tracked_embedding = {}
        global previous_recognition_process
        try:
            while True:
                raw_image = process.stdout.read(frame_size)
                if len(raw_image) == frame_size:
                    yuv = np.frombuffer(raw_image, dtype=np.uint8).reshape((height + height // 2, width))
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    frame = bgr

                    overlay = bgr.copy()
                    output = bgr.copy()
                    alpha = 0.5

                    time_elapsed = time.time() - countdown_start
                    remaining_time = countdown_duration - int(time_elapsed)


                    if remaining_time <0 and previous_recognition_process:
                        countdown_start = time.time()
                    else:
                        if previous_recognition_process and remaining_time<3:
                            text = "Capturing Photo"
                            cv2.putText(overlay, text,(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,0), 4)
                            cv2.putText(overlay, str(remaining_time + 1),(width//2 - 30,height//2), cv2.FONT_HERSHEY_SIMPLEX, 5,(0,0,0), 10)
                        elif previous_recognition_process is False:
                            text = "Processing Photo"
                            cv2.putText(overlay, text,(width//6,height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5)

                    cv2.addWeighted(overlay,alpha,output,1 - alpha,0,output)

                    if previous_recognition_process:
                        current_time = time.time()
                        if current_time - last_saved_time >= 4:
                            last_saved_time = current_time
                            if previous_recognition_process is True:
                                previous_recognition_process = False
                                face_check = executor.submit(process_frame, frame, model_insight, model_antispoof,tracked_embedding)
                                face_check_list.append(face_check)
                    else:
                        countdown_start = time.time()
                        last_saved_time = time.time()

                    cv2.imshow('Frame', output)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        process.terminate()
                        cv2.destroyAllWindows()
                        return HttpResponse(status=505)

                for check in list(face_check_list):
                    if check.done():
                        result = check.result()
                        boolean, matched_label = result
                        print(matched_label)
                        face_check_list.remove(check)
                        if boolean:
                            customer = Customer.objects.get(username=matched_label)
                            record_login(customer)
                            process.terminate()
                            cv2.destroyAllWindows()
                            return HttpResponse(status=200)
                        elif matched_label == "not a member":
                            return JsonResponse({'error': "not a member"}, status=420)
        finally:
            process.terminate()
            cv2.destroyAllWindows()


register_dataset(os.path.abspath('media/customer_photos'),model_insight,embedding_file_path)
