from django.shortcuts import render
import sys
sys.path.append(r'..')
import os
from datetime import datetime
from django.shortcuts import HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.template.context import RequestContext
from django.views.decorators.csrf import csrf_protect

@csrf_exempt
def ajax_test(request):
    user_name = request.POST.get("username")
    password = request.POST.get("password")
    print(user_name, password)
    return HttpResponse("OK")

@csrf_exempt
def ajax_Count(request):
    user_name = request.POST.get("username")
    password = request.POST.get("password")
    print(user_name, password)
    return HttpResponse("OK2")

@csrf_protect
def homepage(request):
    return render(request, 'homepage.html')


@csrf_protect
def diagSearch(request):
    return render(request, 'KnowledgeSearch/diagSearch.html')
    # return render_to_response('KnowledgeSearch/diagSearch.html', context_instance=RequestContext(request))

@csrf_protect
def medSearch(request):
    return render(request, 'KnowledgeSearch/medSearch.html')

@csrf_protect
def prescriptionSearch(request):
    return render(request, 'KnowledgeSearch/prescriptionSearch.html')

@csrf_protect
def patternSearch(request):
    return render(request, 'KnowledgeSearch/patternSearch.html')

@csrf_protect
def index(request):
    return render(request, 'index.html')

@csrf_protect
def drugRisk(request):
    return render(request, 'DrugRisk.html')

# @csrf_protect
# def KnowledgeSearch(request):
#     return render(request, 'KnowledgeSearch')
