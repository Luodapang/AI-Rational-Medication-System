#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import random
import string
import json
import requests
from application import app,get_db
import xlwt as ExcelWrite
import time
import math
from flask import make_response
from io import BytesIO

class TrackService:

    @staticmethod
    def create(no,id,now,type):
        db = get_db()
        # if data['usertype']=='student' :
        #     usertype='STUDENT'
        # elif data['usertype']=='teacher' :
        #     usertype='TEACHER'

        # todo 增加角色标签 Student or Teacher
        results = db.run("MATCH (user:USER{ no : $no }) "
                         "MATCH (venue:VENUE{ id : $id }) "
                         "CREATE (user)-[r:GOTO{time:$time,type:$type}]->(venue) ",
                        {"no": no,"id": id ,"time": now,"type": type}
        )


    @staticmethod
    def createByAdmin(adminno,no,id,now,type):
        db = get_db()
        # if data['usertype']=='student' :
        #     usertype='STUDENT'
        # elif data['usertype']=='teacher' :
        #     usertype='TEACHER'

        # todo 增加角色标签 Student or Teacher
        results = db.run("MATCH (admin:ADMIN{no:$adminno}) - [rr:MANAGE] -> (venue:VENUE{id : $id}) "
                         "MATCH (user:USER{ no : $no }) "
                         "CREATE (user)-[r:GOTO{time:$time,type:$type}]->(venue) ",
                        {"adminno": adminno,"no": no,"id": id ,"time": now,"type": type}
        )


    @staticmethod
    def hasPermission(no,id):
        db = get_db()
        # if data['usertype']=='student' :
        #     usertype='STUDENT'
        # elif data['usertype']=='teacher' :
        #     usertype='TEACHER'

        # todo 增加角色标签 Student or Teacher
        results = db.run("MATCH (user:USER{ no : $no }) -[r:PERMISSION{active:1}]-> (venue:VENUE{ id : $id })"
                         "RETURN r ",
                        {"no": no,"id": id}
        )

        result=results.single()
        
        if not result:#无权限边
            return False
        else:#有权限边
            return True


    @staticmethod
    def getMyTrackByNo(no,pageNum,pageSize):
        db = get_db()
        offset = (pageNum - 1) * pageSize
        # if usertype==1 :
        #     usertype='STUDENT'
        # elif usertype==2:
        #     usertype='TEACHER'

        #获取列表 todo 增加角色标签
        results = db.run("MATCH (user:USER{ no : $no }) -[r:GOTO]-> (venue:VENUE) "
                 "RETURN venue.name AS place,r.time AS time,r.type AS type ORDER BY r.time DESC skip $offset limit $size ", 
                 {"no": no,"offset":offset,"size":pageSize}
        )

        trackList=[]
        for record in results:
            tmp_data = {
                "place": record['place'],
                "time": record['time'],
                "type": record['type'],
            }
            trackList.append(tmp_data)

        return trackList


        # return trackList.append([Track(record) for record in results])

    @staticmethod
    def getTrackByNo(usertype,no,page,size):
        db = get_db()
        
        if usertype==1 :
            usertype='STUDENT'
        elif usertype==2:
            usertype='TEACHER'
        else:
            usertype='STAFF'

        #获取总数
        results = db.run("MATCH (user:USER:"+usertype+"{no:$no})-[r:GOTO]->(venue:VENUE) "
                 "RETURN COUNT(1) AS count", {"no": no}
        )

        result=results.single()
        totalCount=result['count']

        offset = (page - 1) * size

        #获取列表
        results = db.run("MATCH (user:USER:"+usertype+"{no:$no})-[r:GOTO]->(venue:VENUE) "
                 "RETURN r.time AS time,r.type AS type,venue.name AS place ORDER BY r.time DESC skip $offset limit $size ", {"no": no ,"offset":offset,"size":size}
        )

        trackList=[]
        for record in results:
            tmp_data = {
                "place": record['place'],
                "time": record['time'],
                "type": record['type'],
            }
            trackList.append(tmp_data)

        return totalCount,trackList



    @staticmethod
    def search(adminno,type,venue,no,name,date_from,date_to,page,size):
        db = get_db()
        
        if type==0:
            #获取总数
            if adminno=="":#超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN COUNT(1) AS count", {"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","date_from": date_from,"date_to": date_to}
                )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                         "MATCH (user:USER)-[r:GOTO]->(venue) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN COUNT(1) AS count", {"adminno":adminno,"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","date_from": date_from,"date_to": date_to}
                )            
        else:
            #获取总数
            if adminno=="":#超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN COUNT(1) AS count", {"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","type": type,"date_from": date_from,"date_to": date_to}
                )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                         "MATCH (user:USER)-[r:GOTO]->(venue) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN COUNT(1) AS count", {"adminno":adminno,"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","type": type,"date_from": date_from,"date_to": date_to}
                )

        result=results.single()
        totalCount=result['count']

        offset = (page - 1) * size

        if type==0:
            #获取列表
            if adminno=="":#超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC skip $offset limit $size ", {"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","date_from": date_from,"date_to": date_to,"offset":offset,"size":size}
                )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                          "MATCH (user:USER)-[r:GOTO]->(venue) "
                          "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                          "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC skip $offset limit $size ", {"adminno":adminno,"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","date_from": date_from,"date_to": date_to,"offset":offset,"size":size}
                )
        else:
            #获取列表
            if adminno=="":#超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC skip $offset limit $size ", {"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","type": type,"date_from": date_from,"date_to": date_to,"offset":offset,"size":size}
                )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                         "MATCH (user:USER)-[r:GOTO]->(venue) "
                         "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                         "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC skip $offset limit $size ", {"adminno":adminno,"no": "(?i).*" + no + ".*","name": "(?i).*" + name + ".*","venue": "(?i).*" + venue + ".*","type": type,"date_from": date_from,"date_to": date_to,"offset":offset,"size":size}
                )

        trackList=[]
        for record in results:
            tmp_data = {
                "venue": record['venue'],
                "name": record['name'],
                "no": record['no'],
                "time": record['time'],
                "type": record['type'],
            }
            trackList.append(tmp_data)

        return totalCount,trackList




    @staticmethod
    def searchByNoTime(no,date_from,date_to):
        db = get_db()

        results = db.run("MATCH (user:USER{no:$no})-[r:GOTO]->(venue:VENUE) "
                         "WHERE r.time >= $date_from AND r.time <= $date_to "
                         "RETURN r.time AS time,venue.id AS venueid,venue.name AS venuename ORDER BY r.time", {"no": no,"date_from": date_from,"date_to": date_to}
                )

        # venueList=[]
        # for record in results:
        #     tmp_data = {
        #         "place": record['place'],
        #         "time": record['time'],
        #         "type": record['type'],
        #     }
        #     trackList.append(tmp_data)
        i=0
        venueList=[]
        for record in results:
            i=i+1
            tmp_data = {
                "id": str(i),
                "venueid": record['venueid'],
                "venuename": record['venuename'],
                "time": record['time'],
            }
            venueList.append(tmp_data)

        return venueList



    @staticmethod
    def searchByVenueTime(venueid,date_from,date_to,no):
        db = get_db()

        results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE{id:$id}) "
                         "WHERE r.time >= $date_from AND r.time <= $date_to AND user.no<>$no "
                         "RETURN user.no AS userno,user.name AS name,r.time AS time", {"id": venueid,"date_from": date_from,"date_to": date_to,"no": no}
                )

        userList=[]
        for record in results:
            tmp_data = {
                "no": record['userno'],
                "name": record['name'],
                "time": record['time'],
            }
            userList.append(tmp_data)
        return userList

    @staticmethod
    def exportTrackList(track_list):
        header = ['序号', '场所', '学工号', '姓名', '扫码类型', '时间']
        header_dict = {'序号': '_', '场所': "venue", '学工号': "no", '姓名': "name", '扫码类型': "type", '时间': "time"}
        def _make_style(sheet):
            ''' 样式 '''
            for i in range(len(header)):
                if i != 0: sheet.col(i).width = 30 * 256

            first_style = ExcelWrite.XFStyle()
            al = ExcelWrite.Alignment()
            al.horz = 0x02  # 设置水平居中
            al.vert = 0x01  # 设置垂直居中
            first_header = ExcelWrite.Font()
            first_header.height = 20 * 18
            first_header.bold = True
            first_style.alignment = al
            first_style.font = first_header

            # 表头样式
            header_style = ExcelWrite.XFStyle()
            font_header = ExcelWrite.Font()
            font_header.height = 20 * 15
            font_header.bold = True
            header_style.font = font_header
            borders_header = ExcelWrite.Borders()
            borders_header.left = 2
            borders_header.right = 2
            borders_header.top = 2
            borders_header.bottom = 2
            header_style.borders = borders_header

            # 内容样式
            content_style = ExcelWrite.XFStyle()
            font_content = ExcelWrite.Font()
            font_content.height = 20 * 12
            content_style.font = font_content
            content_style.borders = borders_header

            last_style = first_style
            return first_style, header_style, content_style, last_style

        # def data2excel(track_list):
        #     '''  数据生成excel '''
        #     output = BytesIO()
        #     work_book = ExcelWrite.Workbook(output, {'in_memory': True})
        #     sheet = work_book.add_sheet(f'表1')
        #     first_style, header_style, content_style, last_style = _make_style(sheet)
        #
        #     line = 0
        #     sheet.write_merge(line, line + 1, 0, len(header) - 1, u'轨迹统计', first_style)
        #     line += 2
        #     # 写表头
        #     for i, each_header in enumerate(header):
        #         sheet.write(line, i, each_header, header_style)
        #         i += 1
        #     line += 1
        #     # 写内容
        #     for i, each_data in enumerate(track_list):
        #         for j, key in enumerate(header_dict):
        #             if j == 0:
        #                 sheet.write(line, 0, i+1, content_style)
        #             elif header_dict[key] == "type":
        #                 if each_data[header_dict[key]] == 1:
        #                     sheet.write(line, j, "绿码", content_style)
        #                 elif each_data[header_dict[key]] == 2:
        #                     sheet.write(line, j, "黄码", content_style)
        #             else:
        #                 sheet.write(line, j, each_data[header_dict[key]], content_style)
        #         line += 1
        #     sheet.write_merge(line, line + 1, 0, math.floor((len(header) - 1) / 2), '')
        #     sheet.write_merge(line, line + 1, math.floor((len(header) - 1) / 2) + 1, len(header) - 1,
        #                       f'导出时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', last_style)
        #     sio = BytesIO()  # 将获取的数据在内存中写，有时会用到StringIO()
        #     work_book.save(sio)  # 将文件流保存
        #     sio.seek(0)  # 光标
        #     response = make_response(sio.getvalue())  #
        #     # response.headers['Content-type'] = 'application/vnd.ms-excel'  # 指定返回的类型
        #     response.headers['Transfer-Encoding'] = 'chunked'
        #     response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        #     response.headers['Content-Security-Policy'] = "default-src 'self'"
        #     response.headers['X-Content-Type-Options'] = 'nosniff'
        #     response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        #     response.headers['X-XSS-Protection'] = '1; mode=block'
        #     response.headers['Content-Disposition'] = f'attachment;filename={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.xls'  # 设定用户浏览器显示的保存文件名
        #     return response  # 返回response，浏览器会出现如下效果，如果返回其他，比如None就会只下载不在浏览器提示。
        #
        # return data2excel(track_list)

    @staticmethod
    def searchAll(adminno, type, venue, no, name, date_from, date_to):
        db = get_db()

        if type == 0:
            # 获取总数
            if adminno == "":  # 超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN COUNT(1) AS count",
                                 {"no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "date_from": date_from, "date_to": date_to}
                                 )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                                 "MATCH (user:USER)-[r:GOTO]->(venue) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN COUNT(1) AS count",
                                 {"adminno": adminno, "no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "date_from": date_from, "date_to": date_to}
                                 )
        else:
            # 获取总数
            if adminno == "":  # 超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN COUNT(1) AS count",
                                 {"no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "type": type, "date_from": date_from,
                                  "date_to": date_to}
                                 )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                                 "MATCH (user:USER)-[r:GOTO]->(venue) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN COUNT(1) AS count",
                                 {"adminno": adminno, "no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "type": type, "date_from": date_from,
                                  "date_to": date_to}
                                 )

        result = results.single()
        totalCount = result['count']


        if type == 0:
            # 获取列表
            if adminno == "":  # 超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC",
                                 {"no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "date_from": date_from, "date_to": date_to}
                                 )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                                 "MATCH (user:USER)-[r:GOTO]->(venue) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC",
                                 {"adminno": adminno, "no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "date_from": date_from, "date_to": date_to}
                                 )
        else:
            # 获取列表
            if adminno == "":  # 超级管理员
                results = db.run("MATCH (user:USER)-[r:GOTO]->(venue:VENUE) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC",
                                 {"no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "type": type, "date_from": date_from,
                                  "date_to": date_to}
                                 )
            else:
                results = db.run("MATCH (user:ADMIN{no:$adminno}) - [r:MANAGE] -> (venue:VENUE) with venue "
                                 "MATCH (user:USER)-[r:GOTO]->(venue) "
                                 "WHERE user.no =~ $no AND user.name =~ $name AND venue.name =~ $venue AND r.type=$type AND r.time >= $date_from AND r.time <= $date_to "
                                 "RETURN user.no AS no,user.name AS name,r.time AS time,r.type AS type,venue.name AS venue ORDER BY r.time DESC",
                                 {"adminno": adminno, "no": "(?i).*" + no + ".*", "name": "(?i).*" + name + ".*",
                                  "venue": "(?i).*" + venue + ".*", "type": type, "date_from": date_from,
                                  "date_to": date_to}
                                 )

        trackList = []
        for record in results:
            tmp_data = {
                "venue": record['venue'],
                "name": record['name'],
                "no": record['no'],
                "time": record['time'],
                "type": record['type'],
            }
            trackList.append(tmp_data)

        return totalCount, trackList

