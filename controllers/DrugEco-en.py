import sys
sys.path.append(r'..')
from django.shortcuts import HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
import json

from predictMedRegimen import predictMedRegimen

@csrf_exempt
def evaluateEco(request):
    try:
        patient = json.loads(request.POST.get("patient"))
        diags = json.loads(request.POST.get("diag"))
        meds = json.loads(request.POST.get("prescrpt"))
        resp = {"code": 200, "msg": "success", "data": {}}
        data = patient + [','.join(diags)] + [','.join(meds)]
        data = ['4', '男', '106', '17.5', '急性支气管炎,支气管炎,急性上呼吸道感染', '阿莫西林克拉维酸钾干混悬剂(奥先)_3天,吸入用复方异丙托溴铵溶液(可必特)_1天,吸入用布地奈德混悬液(普米克令舒)_1天']

        drug_predict = predictMedRegimen(data)
        print(3333, drug_predict)
        # demo
        drug_predict = [['盐酸左卡巴斯汀鼻喷雾剂(立复汀)_3天', '双歧杆菌乳杆菌三联活菌片(金双歧)_3天', 'Amoxicillin and Clavulanate Potassium for Oral Suspension_three days'], ['吸入用布地奈德混悬液(普米克令舒)_2天', '口服补液盐iii_1天', 'Combivent_one day']]
        resp['data'] = {"Druglist": drug_predict}

    except Exception as ee:
        resp = {"code": 500, "msg": "error3", "data": str(ee)}
        print(resp)
    return HttpResponse(json.dumps(resp), content_type='application/json')