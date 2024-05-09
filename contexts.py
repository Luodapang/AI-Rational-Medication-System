from django.conf import settings
from neo4j.v1 import GraphDatabase

def loadInfo(request):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","neo4j"))
    # driver = GraphDatabase.driver("bolt://it.ye-soft.com:7687", auth=("neo4j","neo4j"))
    db = driver.session();
    results = db.run("MATCH (m:med) RETURN m.name as med")
    Meds = []
    for record in results:
        Meds.append(record['med'])
    # 疾病列表
    results = db.run("MATCH (d:diag) RETURN d.name as diag")
    Diags = []
    for record in results:
        Diags.append(record['diag'])
    return {"Meds":Meds,"Diags":Diags}


def lang(request):
    return {'lang': settings.LANGUAGE_CODE}