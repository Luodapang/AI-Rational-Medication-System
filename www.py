#!/usr/bin/env python
# -*- coding: utf-8 -*-
from application import app

"""
统计拦截器
"""
# from interceptors.Authinterceptor import *
# from interceptors.Errorinterceptor import *

"""
蓝图功能，对所有的 url 进行蓝图功能配置
"""

from controllers.DrugRisk import route_DrugRisk
app.register_blueprint(route_DrugRisk, url_prefix="/DrugRisk")

