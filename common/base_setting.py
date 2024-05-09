#!/usr/bin/env python
# -*- coding: utf-8 -*-
SERVER_PORT = 9000
# DEBUG = False
DEBUG = True

#使用CDN加速静态资源
CDN = True

SALT = "46nourJfj2f4341D"

AUTH_COOKIE_NAME = "xmu_health"

# 过滤 url,以下URL不经过权限判断
IGNORE_URLS = [

    #外部操作用户state
    "^/api/setPersonState",
    "^/api/getPersonState",
    "^/api/setLeader",#设置领导
    "^/api/getLeader",#获取领导

    # 静态资源
    "^/static",
    "^/favicon.ico",

    # 后台管理员
    "^/admin/login",
    "^/admin/callback",

    # 微信小程序用户
    "^/wechat/getAllPlace",#待删除
    "^/wechat/callback",
    "^/wechat/login"
]


PAGE_SIZE = 100
PAGE_DISPLAY = 10

UPLOAD = {
    "ext": ["txt", "TXT"],#["jpg", "gif", "bmp", "jpeg", "png"],
    # "prefix_path": "/webs/static/upload/",
    # "prefix_url": "/static/upload/"
    "whitelist_prefix_path": "/files/whitelist/",
    "adminlist_prefix_path": "/files/adminlist/",
    "wxcode_prefix_path": "/webs/static/images/WXcode/",
    "wxcode_prefix_url": "/static/images/WXcode/"
}

APP = {
    "domain": "http://127.0.0.1:9000"
}


# 本地测试
NEO4J = {
    "address": "bolt://127.0.0.1",
    "username": "neo4j",
    "password": "123456"
}

# 线上
# NEO4J = {
#     "address": "bolt://localhost",
#     "username": "neo4j",
#     "password": "Mars@2018"
# }

# 线上指定IP
# NEO4J = {
#     "address": "bolt://219.229.80.92",
#     "username": "neo4j",
#     "password": "Mars@2018"
# }

MYSQL = {
    "address":"172.27.65.10",
    "username":"usr_clb_2_health",
    "password":"iNXAHWgG4M",
    "schema":"db_xmu_daily_health"
}


WECHAT={
    "APPID": "wx3dce8ac520a3f06c",
    "APPKEY": "d618acaffae2b554aca3b3827c0c8b14"
    # "ACCESS_TOKEN": "32_327CnO6-Z4MMLU8rLmE1uQ4vL_C0X9-RcS--wbgyn0q_5DeIsrUbOs_DtcOLQjIyaF5maZjQt8TooEF7MaFux0e1rHcp7O4elfF4B0C9bh9-wO0Au1rC7M9fbDfadybmLvQB-glLvAEqAmUSNODeAEACDS"
}


TOKEN="DKp5UpBSlTp5J2Ocd1vCquVsnRDcORfAsP1eRSQ5WMbFeYbyourDKuOxoAAOmhPQ"




ALLOWED_IP = [
    "127.0.0.1",
    "210.34.218.170",
    "121.192.190.97",
    "211.97.104.44",#my福州
    "172.18.61.130"#my厦门vpn
]


FILESERVER = {
    "ip":"172.27.65.191",
    "port":22,
    "username":"WXcode",
    "password":"Lsa@173#qw",
    "path":"/home/WXcode/passport_file/WXcode/"
}

