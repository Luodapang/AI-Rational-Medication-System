#!/usr/bin/env python
# -*- coding: utf-8 -*-
from application import get_db
from django.views.decorators.csrf import csrf_exempt
from neo4j.v1 import GraphDatabase
from django.core import serializers

@csrf_exempt

class KnowledgeSearchService:

    @csrf_exempt
    def getDiagSearch(diag, hostAddress):
        # driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
        server_url = "bolt://" + hostAddress + "7687";
        print("server_url：", server_url)
        driver = GraphDatabase.driver(server_url, auth=("neo4j", "neo4j"))
        db = driver.session();
        #数量
        r1 = db.run("match(pre:prescription)-[r]->(d:diag{diag:$diag}) return count(*) as count",
                         {"diag": diag})
        result = r1.single()
        count = result['count']
        #药物
        r2 = db.run("match(d:diag{diag:$diag})-[r1]-(p:prescription)-[r]-(m:med) return distinct m.medName as m,count(*) as fre order by fre desc limit 5",
                       {"diag":diag}
                       )
        medList=[]
        for record in r2:
            tmp_data = {
                "med": record['m'],
                "frequency": record['fre'],
            }
            medList.append(tmp_data)
        #年龄
        age = db.run("match(d:diag{diag:$diag})-[r1]-(p:patient) return distinct p.age as age,count(*) as score order by age ",
                        {"diag": diag}
        )
        ageList=[]
        for record in age:
            tmp_data = {
                "age": record['age'],
                "frequency": record['score'],
            }
            ageList.append(tmp_data)
        #性别
        sex = db.run(
            "match(d:diag{diag:$diag})-[r1]-(p:patient) return distinct p.sex as sex,count(*) as score order by p.sex ",
            {"diag": diag}
            )
        sexList = []
        for record in sex:
            tmp_data = {
                "sex": record['sex'],
                "score": record['score'],
            }
            sexList.append(tmp_data)
        return count,medList,ageList,sexList

    @csrf_exempt
    def getMedSearch(med, hostAddress):
        server_url = "bolt://" + hostAddress + "7687";
        print("server_url：", server_url)
        driver = GraphDatabase.driver(server_url, auth=("neo4j", "neo4j"))
        db = driver.session();
        # 数量
        results = db.run("match(pre:prescription)-[r]->(m:med{medName:$med}) return count(*) as count",
                         {"med": med})
        result = results.single()
        # print(result['count'])
        count =result['count']
        # 病症
        results = db.run(
            "match(d:diag)-[r1]-(p:prescription)-[r]-(m:med{medName:$med}) return distinct d.diag as diag,count(*) as fre order by fre desc limit 5",
            {"med": med}
        )
        diagList = []
        for record in results:
            diagList.append({
                "diag": record['diag'],
                "frequency": record['fre'],
            })
        # print(diagList,type(diagList))
        # 适应症
        results = db.run(
            "match (m:med{medName:$m})-[r:indication]-(d:diag) return m.medName as med,r,d.diag as diag",
            {"m": med}
        )
        indicationList = []
        for record in results:
            indicationList.append(record['diag'])
        # print(indicationList)
        # 成分
        results = db.run(
            "match (m:med{medName:$m})-[r:contain]-(i:ingredient) return m.medName as med,r,i.name as i",
            {"m": med}
        )
        ingredientsList = []
        for record in results:
            ingredientsList.append(record['i'])
        # print(ingredientsList)
        # 禁忌
        results = db.run(
            "match (m:med{medName:$m})-[r:ban]-(m2:med) return m.medName as m1,m2.medName as m2",
            {"m": med}
        )
        banmedsList = []
        for record in results:
            banmedsList.append(record['，m2'])
        # print(banmedsList)
        results = db.run(
            "match (m:med{medName:$m})-[r:ban]-(d:diag) return m.medName as m1,d.diag as d",
            {"m": med}
        )
        bandiagsList = []
        for record in results:
            bandiagsList.append(record['，d'])
        # print(bandiagsList)
        results = db.run(
            "match (m:med{medName:$m})-[r:ban]-(p:patient) return p.name as p",
            {"m": med}
        )
        banpatientsList = []
        for record in results:
            banpatientsList.append(record['p'])
        # print(banpatientsList)
        return count, diagList, ingredientsList, indicationList,banmedsList,bandiagsList,banpatientsList

    @csrf_exempt
    def getprescriptionSearch(diag, hostAddress):
        server_url = "bolt://" + hostAddress + "7687";
        print("server_url：", server_url)
        driver = GraphDatabase.driver(server_url, auth=("neo4j", "neo4j"))
        db = driver.session();
        # 数量
        results = db.run("match (p:prescription)-[r2]-(d:diag{diag:$diag}) return count(*) as count",
                         {"diag": diag})
        result = results.single()
        count = result['count']
        # 常用药
        FreqmedList = []
        results = db.run(
            "match rr=(m:med)-[r1]-(p:prescription)-[r2]-(d:diag{diag:$diag}) return distinct m.medName as med,count(*) as time order by time desc limit 10",
            {"diag": diag})
        for result in results:
            FreqmedList.append({
                "med": result['med'],
                "fre": result['time']
            })
        print(FreqmedList)
        # 年龄
        age = db.run(
            "match rr=(pt:patient)-[r1]-(p:prescription)-[r2]-(d:diag{diag:$diag}) return distinct pt.age as age, count(*) as count order by count desc",
            {"diag": diag}
            )
        ageList = []
        for record in age:
            tmp_data = {
                "age": record['age'],
                "count": record['count'],
            }
            ageList.append(tmp_data)
        # 性别
        sex = db.run(
            "match rr=(pt:patient)-[r1]-(p:prescription)-[r2]-(d:diag{diag:$diag}) return distinct pt.sex as sex, count(*) as count order by count desc",
            {"diag": diag}
        )
        sexList = []
        for record in sex:
            tmp_data = {
                "sex": record['sex'],
                "count": record['count'],
            }
            sexList.append(tmp_data)
        return count, FreqmedList, ageList, sexList

    @csrf_exempt
    def getPatternSearch(med, hostAddress):
        server_url = "bolt://" + hostAddress + "7687";
        print("server_url：", server_url)
        driver = GraphDatabase.driver(server_url, auth=("neo4j", "neo4j"))
        db = get_db()
        results = db.run(
            "match(m1:med{medName:$med}) match(m2:med) return distinct m2.medName as med,algo.linkprediction.commonNeighbors(m1,m2) as fre order by fre desc limit 10",
            {"med": med}
        )
        coMedList = []
        for record in results:
            tmp_data = {
                "med": record['med'],
                "frequency": record['fre'],
            }
            coMedList.append(tmp_data)
        return coMedList
