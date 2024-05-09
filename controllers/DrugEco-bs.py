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
        # "5\t男\t106\t17.5\t急性支气管炎,支气管炎,急性上呼吸道感染\t阿莫西林克拉维酸钾干混悬剂(奥先)_1包_3天,吸入用复方异丙托溴铵溶液(可必特)_2.5ml_1天,吸入用布地奈德混悬液(普米克令舒)_2.0ml_3天,布地奈德鼻喷雾剂(雷诺考特)_1.0喷_3天",  // 错误
        # drug_predict = [['_', '_', '阿莫西林克拉维酸钾干混悬剂(奥先)_1包_3天'], ['_', '_', '吸入用复方异丙托溴铵溶液(可必特)_1.25ml_1天'], ['_', '_', '吸入用布地奈德混悬液(普米克令舒)_2.0ml_1天']]
        drug_predict = [['_', '_', '阿莫西林克拉维酸钾干混悬剂_1包_3天'], ['_', '_', '吸入用复方异丙托溴铵溶液_1.25ml_1天'], ['_', '_', '吸入用布地奈德混悬液_2.0ml_1天']]
        resp['data'] = {"Druglist": drug_predict}

    except Exception as ee:
        resp = {"code": 500, "msg": "error3", "data": str(ee)}
        print(resp)
    return HttpResponse(json.dumps(resp), content_type='application/json')