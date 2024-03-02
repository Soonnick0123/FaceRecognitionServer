from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,JsonResponse

@csrf_exempt
def hello_world ( request ):
    return JsonResponse({ 'message': 'Hello, world!' })

@csrf_exempt
def second_function(request):
    return JsonResponse({'message': "Second Message!"})