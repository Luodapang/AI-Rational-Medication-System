

function to_page(pageNum) {

    var data = {

        diag: $('#diagInput').val(),
    };

    $.ajax({
        url: common_ops.buildUrl("/track/getTrackList"),
        data: data,
        type: "POST",
        dataType: "json",
        success: function (data) {
            //解析并显示数据表
            build_table(data);
            //解析并显示分页数据
            build_page_nav(data);
        }, fail: function (err) {
            console.log(err);
        }
    })

    function build_table(data) {
        //清空table表格
        $("#track_table tbody").empty();
        var track = data.list;
        // console.log(users.length);
        if (track.length == 0) {
            var item = $("<td></td>").attr("colspan", "6").append("暂无数据");
            $("<tr></tr>").append(item).appendTo("#track_table tbody");
            return;
        }
        var offset = (data.pages.current - 1) * data.pages.page_size
        //遍历元素
        $.each(track, function (index, item) {

            var seq = $("<td></td>").append(index + 1 + offset);
            var venue = $("<td onclick=clickTd(this) id='venue' style='text-decoration: underline;color:#3975A6;cursor:pointer;'></td>").append(item.venue);
            var no = $("<td onclick=clickTd(this) id='no' style='text-decoration: underline;color:#3975A6;cursor:pointer;'></td>").append(item.no);
            var name = $("<td onclick=clickTd(this) id='name' style='text-decoration: underline;color:#3975A6;cursor:pointer;'></td>").append(item.name);

            if (item.type == 1)
                var type = $("<td onclick=clickTd(this) id='type' value='1' style='cursor:pointer;'></td>").append("<div style='width:20px;height:20px;background-color:#36B44C'></div>");
            if (item.type == 2)
                var type = $("<td onclick=clickTd(this) id='type' value='2' style='cursor:pointer;'></td>").append("<div style='width:20px;height:20px;background-color:#FABC18'></div>");

            var time = $("<td></td>").append(item.time);

            $("<tr></tr>").append(seq).append(venue).append(no).append(name).append(type).append(time).appendTo("#track_table tbody");

        })
    }

    function build_page_nav(data) {
        var pages = data.pages;
        $("#pagination .col-lg-12 span").empty();
        $("#pagination .col-lg-12 span").append("共" + pages.total + "条记录 | 每页" + pages.page_size + "条");
        var dom = $("#pagination .col-lg-12 ul");
        dom.empty();

        if (pages.is_prev == 1) {
            var itemFirst = $("<li></li>").append(($("<a></a>")).attr("href", "#").append(($("<span></span>")).append("首页")));

            itemFirst.click(function () {
                to_page(1);
            });

            dom.append(itemFirst);
        }


        var range = [];
        for (var i = pages.from; i < pages.end + 1; i++) {
            range = range.concat(i);
        }

        $.each(range, function (index, item) {
            if (item == pages.current) {
                var temp = $("<li></li>").addClass("active").append($("<a></a>").attr("href", "#").append(item));
                dom.append(temp);
            } else {
                var temp = $("<li></li>").append($("<a></a>").attr("href", "#").append(item));

                temp.click(function () {
                    to_page(item);
                });
                dom.append(temp);
            }
        });

        if (pages.is_next == 1) {
            var itemLast = $("<li></li>").append(($("<a></a>")).attr("href", "#").append(($("<span></span>")).append("尾页")));

            itemLast.click(function () {
                to_page(pages.total_pages);
            });

            dom.append(itemLast);
        }
    }
}


//搜索
$(".wrap_search .search").click(function () {
    to_page(1);
});


