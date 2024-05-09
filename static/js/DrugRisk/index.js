

    // var token = $('input[name=csrfmiddlewaretoken]').val();
    // function search(){
    //     var diag = document.getElementById("diagInput").value
    //     draw(diag);
    //     loadCount(diag);
    // }
    //
    // function loadCount(diag) {
    //     var data = {
    //         diag:diag,
    //         csrfmiddlewaretoken:$('[name="csrfmiddlewaretoken"]').val()
    //     };
    //     console.log(data);
    //     $.ajax({
    //       url: "/KnowledgeSearch/diagSearch/",
    //       type: "POST",
    //       data: data,
    //       success: function (data) {
    //         console.log(JSON.stringify(data))
    //
    //       }
    //     })
    // }
    // function build_table(){
    //
    //     $("#track_table tbody").empty();
    //     var track = data.list;
    //     // console.log(users.length);
    //     if (track.length == 0) {
    //         var item = $("<td></td>").attr("colspan", "6").append("暂无数据");
    //         $("<tr></tr>").append(item).appendTo("#track_table tbody");
    //         return;
    //     }
    //     var offset = (data.pages.current - 1) * data.pages.page_size
    //     //遍历元素
    //     $.each(track, function (index, item) {
    //
    //         var seq = $("<td></td>").append(index + 1 + offset);
    //         var venue = $("<td onclick=clickTd(this) id='venue' style='text-decoration: underline;color:#3975A6;cursor:pointer;'></td>").append(item.venue);
    //         var no = $("<td onclick=clickTd(this) id='no' style='text-decoration: underline;color:#3975A6;cursor:pointer;'></td>").append(item.no);
    //         var name = $("<td onclick=clickTd(this) id='name' style='text-decoration: underline;color:#3975A6;cursor:pointer;'></td>").append(item.name);
    //
    //         if (item.type == 1)
    //             var type = $("<td onclick=clickTd(this) id='type' value='1' style='cursor:pointer;'></td>").append("<div style='width:20px;height:20px;background-color:#36B44C'></div>");
    //         if (item.type == 2)
    //             var type = $("<td onclick=clickTd(this) id='type' value='2' style='cursor:pointer;'></td>").append("<div style='width:20px;height:20px;background-color:#FABC18'></div>");
    //
    //         var time = $("<td></td>").append(item.time);
    //
    //         $("<tr></tr>").append(seq).append(venue).append(no).append(name).append(type).append(time).appendTo("#track_table tbody");
    //
    //     })
    // }
    //
    //
    // var diagSearchviz;
    // function draw(diag) {
    //     console.log("diag:",diag)
    //     if (diag == null || diag == "")
    //         cypher = "match (d:diag{name:\"休克\"})-[r]-(m) return d,r,m"
    //     else
    //         cypher = "match (d:diag{name:\""+diag+"\"})-[r]-(m) return d,r,m"
    //     var config = {
    //         container_id: "diagSearchviz",
    //         server_url: "bolt://localhost:7687",
    //         server_user: "neo4j",
    //         server_password: "sorts-swims-burglaries",
    //         labels: {
    //             "patient": {
    //                 "caption": "pid",
    //                 "age": "age",
    //                 "height": "height",
    //                 "sex": "sex",
    //                 "weight":"weight",
    //                 //"sizeCypher": "MATCH (n) WHERE id(n) = {id} MATCH (n)-[r]-() RETURN sum(r.weight) AS c"
    //             },
    //             "med": {
    //                 "caption": "medName",
    //                 "medSpec": "medSpec",
    //             }
    //         },
    //         relationships: {
    //             "huse": {
    //                 "thickness": "weight",
    //                 "caption": false
    //             }
    //         },
    //         initial_cypher: cypher
    //     };
    //
    //     neo4jd3viz = new NeoVis.default(config);
    //     neo4jd3viz.render();
    //     console.log(neo4jd3viz);
    //
    // }
    // window.onload = draw();


// window.onload = draw();
// function visualize() {
//     draw();
//     console.log("visualize")
//     // resp_data = {"code": 200, "msg": "操作成功", "data": {}}
//     // var nodesList=[{"id": "001","labels": ["med"],"properties":{"medname":"清开灵"},},{"id": "002","labels": ["med"],"properties":{"medname":"头孢拉定片"}}]
//     // var relationshipList = [{"type": "interaction","startNode": "001","endNode": "002","properties": {"类型": "抑制"}}]
//     // nodesList.append({
//     //         "id": "001",
//     //         "labels": ["med"],
//     //         "properties":{"medname":"清开灵"},
//     //     })
//     // nodesList.append({
//     //         "id": "002",
//     //         "labels": ["med"],
//     //         "properties":{"medname":"头孢拉定片"},
//     //     })
//     // relationshipList.append({
//     //             "type": "interaction",
//     //             "startNode": "001",
//     //             "endNode": "002",
//     //             "properties": {
//     //                 "类型": "抑制"
//     //             }
//     //         })
//     // defaultjsonData={"results":[{"data":[{"graph":{"nodes":nodesList,"relationships":relationshipList}}]}]};
//     // resp_data["data"] = defaultjsonData;
//     // // var defaultData = jsonify(resp_data)
//     // var defaultData = JSON.parse(resp_data)
//     // window.onload = init(defaultData.data);
//     window.onload = init();
//     var data = {
//         no:"2302019001"
//     };
//
//     $.ajax({
//         url: "/DrugRisk/getMeds",
//         data:data,
//         type: "POST",
//         dataType: "json",
//         success: function (data) {
//             window.onload = init(data.data);
//             console.log(data)
//             //解析并显示数据表
//             // build_table(data);
//         },fail:function(err){
//             console.log(err);
//         }
//     })
//
//     function init(data) {
//
//         var neo4jd3 = new Neo4jd3('#neo4jd3', {
//             images: {
//                 // '场所': "https://passport.xmu.edu.cn/static/images/common/venue.svg",
//                 // '目标人员': "https://passport.xmu.edu.cn/static/images/common/riskuser.svg",
//                 // '人员': "https://passport.xmu.edu.cn/static/images/common/user.svg",
//             },
//             minCollision: 60,
//             neo4jData:data,
//
//             nodeRadius: 25,
//             // onNodeDoubleClick: function(node) {
//             //     switch(node.id) {
//             //         case '25':
//             //             // Google
//             //             window.open(node.properties.url, '_blank');
//             //             break;
//             //         default:
//             //             var maxNodes = 5,
//             //                 data = neo4jd3.randomD3Data(node, maxNodes);
//             //             neo4jd3.updateWithD3Data(data);
//             //             break;
//             //     }
//             // },
//             // onRelationshipDoubleClick: function(relationship) {
//             //     console.log('double click on relationship: ' + JSON.stringify(relationship));
//             // },
//             // zoomFit: true
//             });
//     }
//
//
//
//
//
//     function build_table(data) {
//         //清空table表格
//         $("#track_table tbody").empty();
//         var track = data.list;
//         // console.log(users.length);
//         if(track.length==0){
//             var item=$("<td></td>").attr("colspan","6").append("暂无数据");
//             $("<tr></tr>").append(item).appendTo("#track_table tbody");
//             return;
//         }
//         var offset = (data.pages.current-1)*data.pages.page_size
//         //遍历元素
//         $.each(track, function (index, item) {
//
//             var seq = $("<td></td>").append(index+1+offset);
//             var venue = $("<td></td>").append(item.venue);
//             var no = $("<td></td>").append(item.no);
//             var name = $("<td></td>").append(item.name);
//
//             if(item.type==1)
//                 var type = $("<td></td>").append("<div style='width:20px;height:20px;background-color:#36B44C'></div>");
//             if(item.type==2)
//                 var type = $("<td></td>").append("<div style='width:20px;height:20px;background-color:#FABC18'></div>");
//
//             var time = $("<td></td>").append(item.time);
//
//             $("<tr></tr>").append(seq).append(venue).append(no).append(name).append(type).append(time).appendTo("#track_table tbody");
//
//         })
//     }
// }


//搜索
// $(".prescription .search").click(function () {
//     console.log("SEARCH")
//     visualize();
//     console.log("SEARCH")
// });
