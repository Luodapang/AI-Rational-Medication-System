from django.shortcuts import render
import sys
sys.path.append(r'..')
import os
from datetime import datetime
from django.shortcuts import HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt, csrf_protect


@csrf_protect
def DrugEco(request):
    return render(request, 'DrugEco.html')