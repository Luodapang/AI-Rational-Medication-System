from common.libs.KnowledgeSearchService import KnowledgeSearchService
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.core import serializers
import json
from neo4j.v1 import GraphDatabase
from django.shortcuts import render
from django.http import JsonResponse

@csrf_exempt
def diagSearch(request):
    try:
        diag = request.POST.get("diag")
        hostAddress = request.POST.get("hostAddress")
        print(diag, type(diag))
        resp = {"code": 200, "msg": "success", "data": {}}
        count, medList, ageList, sexList = KnowledgeSearchService.getDiagSearch(diag, hostAddress)
        print(count,medList,ageList,sexList)
        resp['data']={"count":count,
                      "medList":medList,
                      "ageList":ageList,
                      "sexList":sexList}
    except Exception as e:
        resp = {"code": 500, "msg": "error4", "data": str(e)}
        print(resp)
    return HttpResponse(json.dumps(resp), content_type='application/json')

@csrf_exempt
def medSearch(request):
    try:
        med = request.POST.get("med")
        hostAddress = request.POST.get("hostAddress")
        print(med,type(med))
        resp = {"code": 200, "msg": "success", "data": {}}
        count, diagList, ingredientsList, indicationList,banmedsList,bandiagsList,banpatientsList = KnowledgeSearchService.getMedSearch(med, hostAddress)
        print(count,diagList, ingredientsList, indicationList,banmedsList,bandiagsList,banpatientsList)
        resp['data']={"count":count,
                      "diagList":diagList,
                      "ingredientsList":ingredientsList,
                      "indicationList":indicationList,
                      "banmedsList":banmedsList,
                      "bandiagsList":bandiagsList,
                      "banpatientsList":banpatientsList}
    except Exception as e:
        resp = {"code": 500, "msg": "error5", "data": str(e)}
        print(resp)
    return HttpResponse(json.dumps(resp),content_type='application/json')

@csrf_exempt
def prescriptionSearch(request):
    try:
        diag = request.POST.get("diag")
        hostAddress = request.POST.get("hostAddress")
        print(diag, type(diag))
        resp = {"code": 200, "msg": "success", "data": {}}
        count,FreqmedList,ageList,sexList = KnowledgeSearchService.getprescriptionSearch(diag, hostAddress)
        print(count,FreqmedList,ageList,sexList)
        resp['data']={"count":count,
                      "FreqmedList":FreqmedList,
                      "ageList":ageList,
                      "sexList":sexList}
    except Exception as e:
        resp = {"code": 500, "msg": "error6", "data": str(e)}
        print(resp)
    return HttpResponse(json.dumps(resp),content_type='application/json')
@csrf_exempt
def patternSearch(request):
    try:
        med = request.POST.get("med")
        hostAddress = request.POST.get("hostAddress")
        print(med,type(med))
        resp = {"code": 200, "msg": "success", "data": {}}
        coMedList = KnowledgeSearchService.getPatternSearch(med, hostAddress)
        print(coMedList)
        resp['data']={"coMedList":coMedList,}
    except Exception as e:
        resp = {"code": 500, "msg": "error7", "data": str(e)}
        print(resp)
    return HttpResponse(json.dumps(resp),content_type='application/json')


