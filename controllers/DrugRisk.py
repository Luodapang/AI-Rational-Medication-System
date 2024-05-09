# -*- coding: utf-8 -*-
# from common.libs.Helper import ops_render, iPagination
import json
from neo4j.v1 import GraphDatabase
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
e = 2.36


@csrf_exempt
def getInfo(request):
    resp = {"code": 200, "msg": "success", "data": {}}
    # 连接neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
    db = driver.session();
    results = db.run("MATCH (m:med) RETURN m.medName as med")
    Meds = []
    for record in results:
        Meds.append(record['med'])
    # 疾病列表
    results = db.run("MATCH (d:diag) RETURN d.diag as diag")
    Diags = []
    for record in results:
        Diags.append(record['diag'])
    print(Meds)
    print(Diags)
    resp['data']={"Meds":Meds,
                  "Diags":Diags}
    return HttpResponse(json.dumps(resp),content_type='application/json')
    # return HttpResponse("OKK")


@csrf_exempt
def evaluateRisk(request):
    try:
        patient = json.loads(request.POST.get("patient"))
        diags = json.loads(request.POST.get("diag"))
        meds = json.loads(request.POST.get("prescrpt"))
        resp = {"code": 200, "msg": "success", "data": {}}
        # 对每种药遍历禁忌
        # 判断人群：孕妇、老人、儿童
        e = 2.714
        RDDI = 0
        RcoUse = 0
        RDMF = 0
        RDM = 0
        CRR = 0
        Rgroup = 0
        Rgrouplist = []
        coMedList = []
        YBDZList = []
        RareMedList = []
        RarecoUselist = []
        DoseRisklist = []
        DDIrisklist = []
        CustomRiskList = []
        riskevaluate = 0;
        role = []
        # 连接neo4j
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
        db = driver.session();
        if int(patient[1]) > 55:
            role.append("老人");
        if int(patient[1]) < 14:
            role.append("儿童");
        if int(patient[1]) < 5:
            role.append("幼儿");
        if int(patient[1]) < 2:
            role.append("新生儿");
        for i in range(len(diags)):
            if str(diags[i]).find("孕")!=-1 or str(diags[i]).find("产")!=-1 or str(diags[i]).find("妊娠")!=-1:
                role.append("孕妇");
        print("角色")
        print(role)
        print(meds)
        print(diags)
        print(patient)
        # L_1= Max(e^(degree(i,j)-t)*(1-f_(i,j)))  需要遍历两两药物对 1.读取风险级别  2.计算频率（联用次数/包含药品的处方数）
        for k in range(0,len(meds)):
            YBDZ_flag = 0;
            print("药-病")
            #找出药的所有禁忌，看处方中是否匹配
            resultsC = db.run("match(m:med{medName:$med})-[r: ban]->(d) return distinct m.medName as med, r.reason as r, d as d",{"med":meds[k]})
            for recordC in resultsC:
                print(recordC['r'])
                if str(recordC['r']) == "禁忌症":
                    print("禁忌症")
                    print(str(recordC['d']['diag']))
                    for d in diags:
                        if str(recordC['d']['diag']) in d:
                            CRR += e;
                            customRisk = str(d) + " 为 " + meds[k] + "的 <b>禁忌症</b> ";
                            print(customRisk);
                            if customRisk not in CustomRiskList:
                                CustomRiskList.append(customRisk);
                            YBDZ_flag = 1;
                            RDM = RDM + 1;
                            if {"med":meds[k]} not in YBDZList:
                                YBDZList.append({"med":meds[k]});
                if str(recordC['r']) == "禁用人群":
                    print("禁用人群")
                    print(str(recordC['d']['diag']))
                    if str(recordC['d']['name']) in role:
                            CRR += e;
                            customRisk = str(recordC['d']['name']) + " <b>禁用</b> " + meds[k];
                            print(customRisk);
                            if customRisk not in CustomRiskList:
                                CustomRiskList.append(customRisk);
                if str(recordC['r']) == "过敏":
                    print("过敏")
                    print(str(recordC['d']['medName']))
                    if str(recordC['d']['medName']).find(str(patient[4]))!=-1:
                            CRR += e;
                            customRisk = "患者对 " + meds[k] + "的成分 <b>过敏</b>";
                            print(customRisk);
                            if customRisk not in CustomRiskList:
                                CustomRiskList.append(customRisk);
            if YBDZ_flag == 0:     #判断药是否不对症，1为不对症，不对症则为多开错开风险
                for i in range(0, len(diags)):
                #药不对症、罕见用药
                    results3 = db.run("match(d:diag{diag:$diag})-[r]-(m:med{medName:$med}) return type(r)",{"diag":diags[i],"med":meds[k]})
                    for relation in results3:
                        print("relation")
                        print(relation)
                        if relation == "indicaiton":
                            YBDZ_flag = 0
                        else:
                            results3 = db.run("match(d:diag{diag:$diag})-[r]-(p:prescription)-[r2]-(m:med{medName:$med}) return count(*) as count,m.medName as med",{"diag":diags[i],"med":meds[k]})
                            for record3 in results3:
                                # if int(record3['count'])!=0:
                                if int(record3['count']) == 1 :
                                    if {"med":record3['med']} not in RareMedList:
                                        RareMedList.append({"med":record3['med'],"frequency":record3['count']})
                                # elif int(record3['count']) <= 10:
                                if int(record3['count']) <= 4 and int(record3['count']) >2:
                                    RDMF = RDMF + 1/(e+int(record3['count']))
        print("药-药")
        for i in range(0,len(meds)):
            for j in range(i+1,len(meds)):
                if (i!=j):
                # L1药物联用
                    results = db.run(
                        "match(m1:med{medName:$m1}) match(m2:med{medName:$m2}) return distinct m1.medName as m1, m2.medName as m2,algo.linkprediction.commonNeighbors(m1,m2) as fre order by fre desc limit 10",
                        {"m1": meds[i],"m2":meds[j]}
                    )
                    for record in results:
                        tmp_data = {
                            "m1": record['m1'],
                            "m2": record['m2'],
                            "frequency": record['fre'],
                        }
                        coMedList.append(tmp_data)
                        if (int(record['fre'])):
                            RcoUse = RcoUse + 1/(pow(int(record['fre']),2)+1) - 1/e
                        if (int(record['fre'])<5):
                                RarecoUselist.append({"med1":record['m1'],
                                                  "med2":record['m2'],
                                                 "frequency":record['fre']})
                # L2 DDI
                    results2 = db.run("match(m1:med{medName:$m1})-[r]-(m2:med{medName:$m2}) return m1.medName as m1,m2.medName as m2, r.weight as risk",{"m1":meds[i],"m2":meds[j]})
                    for record2 in results2:
                        if int(record2['risk']) >= 2:
                            tmp_data = {
                                "m1": record2['m1'],
                                "m2": record2['m2'],
                                "risk": record2['risk'],
                            }
                            DDIrisklist.append(tmp_data)
                            RDDI = RDDI + pow(e,int(record2['risk'])-4)
            # results5 = db.run("match(m:med{medName:$med}) return m.community as community",{"med":meds[i]})
            # for record5 in results5:
            #     if int(record5['community']) not in Rgrouplist:
            #         Rgrouplist.append(int(record5['community']))
        Rgroup = len(Rgrouplist)
        riskevaluate = RDDI + RcoUse + RDM/e + RDMF/e + Rgroup/len(meds) + CRR;
        # print("1");
        print(DDIrisklist, RarecoUselist,YBDZList,DoseRisklist,CustomRiskList)
        # print("2");
        print(RDDI, RcoUse,RDM/e,RDMF,Rgroup/len(meds),CRR,riskevaluate);
        if riskevaluate>e:
            print("危险")
            evaluateResult = 2
        elif riskevaluate<1:
            print("安全")
            evaluateResult = 0
        else:
            print("警示")
            evaluateResult = 1
        resp['data']={"RarecoUselist":RarecoUselist,
                      "DDIrisklist":DDIrisklist,
                      "RareMedList":RareMedList,
                      "YBDZList":YBDZList,
                      "DoseRisklist":DoseRisklist,
                      "CustomRiskList":CustomRiskList,
                      "evaluateResult":evaluateResult}
        print(resp['data'])
    except Exception as ee:
        resp = {"code": 500, "msg": "error2", "data": str(ee)}
        print(resp)
    print(resp)
    return HttpResponse(json.dumps(resp),content_type='application/json')
    # return evaluateResult
# @csrf_exempt
# def evaluateRisk(request):
#     diags = json.loads(request.POST.get("diag"))
#     meds = json.loads(request.POST.get("prescrpt"))
#     resp = {"code": 200, "msg": "success", "data": {}}
#     RDDI = 0
#     RcoUse = 0
#     RDMF = 0
#     RDM = 0
#     Rgroup = 0
#     Rgrouplist = []
#     coMedList = []
#     YBDZList = []
#     RareMedList = []
#     RarecoUselist = []
#     DoseRisklist = []
#     DDIrisklist = []
#     riskevaluate = 0;
#     # 连接neo4j
#     driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
#     db = driver.session();
#     for k in range(0,len(meds)):
#         YBDZ_flag = 1     #判断药是否不对症，1为不对症，不对症则为多开错开风险
#         for i in range(0, len(diags)):
#     #药不对症、罕见用药
#             results3 = db.run("match(d:diag{diag:$diag})-[r]-(p:prescription)-[r2]-(m:med{medName:$med}) return count(*) as count,m.name as med",{"diag":diags[i],"med":meds[k]})
#             for record3 in results3:
#                 # if int(record3['count'])!=0:
#                 if int(record3['count'])>1:
#                     YBDZ_flag = 0
#                     if int(record3['count'])<5:
#                         RareMedList.append({"med":record3['med'],"frequency":record3['count']})
#                 # elif int(record3['count']) <= 10:
#                 if int(record3['count']) <= 4 and int(record3['count']) >2:
#                     RDMF = RDMF + 1/(e+int(record3['count']))
#         if YBDZ_flag == 1:
#             RDM = RDM + 1
#             YBDZList.append({"med":meds[k]})
#     for i in range(0,len(meds)):
#         for j in range(i+1,len(meds)):
#             if (i!=j):
#             # L1药物联用
#                 results = db.run(
#                     "match(m1:med{medName:$m1}) match(m2:med{medName:$m2}) return distinct m1.medName as m1, m2.medName as m2,algo.linkprediction.commonNeighbors(m1,m2) as fre order by fre desc limit 10",
#                     {"m1": meds[i],"m2":meds[j]}
#                 )
#                 for record in results:
#                     tmp_data = {
#                         "m1": record['m1'],
#                         "m2": record['m2'],
#                         "frequency": record['fre'],
#                     }
#                     coMedList.append(tmp_data)
#                     if (int(record['fre'])):
#                         RcoUse = RcoUse + 1/(pow(int(record['fre']),2)+1) - 1/e
#                     if (int(record['fre'])<5):
#                         RarecoUselist.append({"med1":record['m1'],
#                                               "med2":record['m2'],
#                                              "frequency":record['fre']})
#             # L2 DDI
#                 results2 = db.run("match(m1:med{medName:$m1})-[r]-(m2:med{medName:$m2}) return r.degree as degree, r.type as risk",{"m1":meds[i],"m2":meds[j]})
#                 for record2 in results2:
#                     tmp_data = {
#                         "m1": record2['m1'],
#                         "m2": record2['m2'],
#                         "risk": record2['risk'],
#                     }
#                     DDIrisklist.append(tmp_data)
#                     RDDI = RDDI + pow(e,int(record2['risk'])-4)
#         results5 = db.run("match(m:med{medName:$med}) return m.community as community",{"med":meds[i]})
#         for record5 in results5:
#             if int(record5['community']) not in Rgrouplist:
#                 Rgrouplist.append(int(record5['community']))
#     Rgroup = len(Rgrouplist)
#     riskevaluate = RDDI + RcoUse + RDM/e + RDMF/e + Rgroup/len(meds)
#     print(RDDI, RcoUse,RDM/e,RDMF,Rgroup/len(meds),riskevaluate)
#     if riskevaluate>e:
#         print("危险")
#         evaluateResult = 2
#     elif riskevaluate<1:
#         print("安全")
#         evaluateResult = 0
#     else:
#         print("警示")
#         evaluateResult = 1
#     resp['data']={"RarecoUselist":RarecoUselist,
#                   "DDIrisklist":DDIrisklist,
#                   "RareMedList":RareMedList,
#                   "YBDZList":YBDZList,
#                   "DoseRisklist":DoseRisklist,
#                   "evaluateResult":evaluateResult}
#     print(resp['data'])
#     return HttpResponse(json.dumps(resp),content_type='application/json')
#     # return HttpResponse("OKK")
#
#
#     # print(presc,type(presc))
#     # resp = {"code": 200, "msg": "success", "data": {}}
#     # # 连接neo4j
#     # driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
#     # db = driver.session();
#     # for m in presc:
#     #     for m2
#     # results = db.run(
#     #     "match(m1:med{name:$m1}) match(m2:med{name:$m2}) return distinct m1.name as m1, m2.name as m2,algo.linkprediction.commonNeighbors(m1,m2) as fre order by fre desc limit 10",
#     #     {"m1": m1,"m2":m2}
#     # )
#     # coMedList = []
#     # for record in results:
#     #     tmp_data = {
#     #         "m1": record['m1'],
#     #         "m2": record['m2'],
#     #         "frequency": record['fre'],
#     #     }
#     #     coMedList.append(tmp_data)
#     #
#     # print(coMedList, type(coMedList))
#     # resp['data']={"coMedList":coMedList}
#     # return HttpResponse(json.dumps(resp),content_type='application/json')
#     # return HttpResponse("OKK")


@csrf_exempt
def matchDiag(request):
    resp = {"code": 200, "msg": "success", "data": {}}
    input = request.POST.get("inputdata")
    Meds,Diags = getInfo();
    print(Diags);
    ouputList = ["123","456"];
    resp['data']={"ouput":ouputList}
    return HttpResponse(json.dumps(resp),content_type='application/json')
    # return HttpResponse("OKK")

