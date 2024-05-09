#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from flask import Flask,g
# from flask_script import Manager
import os
# from common.libs.UrlManager import UrlManager
from neo4j.v1 import GraphDatabase

# class Application(Flask):
#     def __init__(self, import_name, template_folder=None, root_path=None):
#         super(Application, self).__init__(import_name, template_folder=template_folder, root_path=root_path,
#                                           static_folder=None)
#
#         self.config.from_pyfile("config/base_setting.py")
#
# app = Application(__name__, template_folder=os.getcwd() + "/templates", root_path=os.getcwd())

# driver = GraphDatabase.driver(app.config.get("NEO4J")["address"],auth=basic_auth(app.config.get("NEO4J")["username"], app.config.get("NEO4J")["password"]))
try:
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
    # driver = GraphDatabase.driver("bolt://it.ye-soft.com:7687", auth=("neo4j", "neo4j"))
except Exception as e:
        print(f'错误类型10：{e}')
def get_db():
    return driver.session()
    # if not hasattr(g, 'neo4j_db'):
    #     g.neo4j_db = driver.session()
    # return g.neo4j_db

# @app.teardown_appcontext
def close_db(error):
    driver.session.close();
    # if hasattr(g, 'neo4j_db'):
    #     g.neo4j_db.close()



# manager = Manager(app)
#
# '''
# 函数模板
# '''
# app.add_template_global(UrlManager.buildStaticUrl, "buildStaticUrl")
# app.add_template_global(UrlManager.buildUrl, "buildUrl")
# app.add_template_global(UrlManager.buildImageUrl, "buildImageUrl")
