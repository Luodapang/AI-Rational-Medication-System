/*!
 * ====================================================

 * ====================================================
 */
(function(){
	var oldData;
	var html = '';
	html += '<a class="diy export" data-type="json" style="display: none">导出json</a>',
	//html += '<button class="diy export" data-type="json" type="button" class="btn btn-info">打开说明书</button>',

	$('.editor-title').append(html);

	$('.diy').css({
		// 'height': '30px',
		// 'line-height': '30px',
		'margin-top': '0px',
		'float': 'right',
		'background-color': '#fff',
		'min-width': '60px',
		'text-decoration': 'none',
		color: '#999',
		'padding': '0 10px',
		border: 'none',
		'border-right': '1px solid #ccc',
	});
	$('.input').css({
		'overflow': 'hidden',
		'position': 'relative',
	}).find('input').css({
		cursor: 'pointer',
		position: 'absolute',
		top: 0,
		bottom: 0,
		left: 0,
		right: 0,
		display: 'inline-block',
		opacity: 0
	});
	$('.export').css('cursor','not-allowed');

	$(document).on('mouseover', '.export', function(event) {
		// 链接在hover的时候生成对应数据到链接中
		event.preventDefault();
		var $this = $(this),
				type = $this.data('type'),
				exportType;
		switch(type){
			case 'km':
				exportType = 'json';
				break;
			case 'md':
				exportType = 'markdown';
				break;
			default:
				exportType = type;
				break;
		}
		if(JSON.stringify(oldData) == JSON.stringify(editor.minder.exportJson())){
			return;
		}else{
			oldData = editor.minder.exportJson();
		}

		editor.minder.exportData(exportType).then(function(content){
			switch(exportType){
				case 'json':
					console.log($.parseJSON(content));
					break;
				default:
					console.log(content);
					break;
			}
			$this.css('cursor', 'pointer');
			var blob = new Blob([content]),
					url = URL.createObjectURL(blob);
			var aLink = $this[0];
			aLink.href = url;
			aLink.download = $('#node_text1').text()+'.'+type;
		});
	});

})();

function exportJ(){
    var type = $(this).data('type'),
            exportType;
            exportType = 'json';
    type = 'json';
    editor.minder.exportData(exportType).then(function(content) {
        if (content != null) {
            switch (exportType) {
                case 'json':
                    console.log($.parseJSON(content));
                    break;
                default:
                    console.log(content);
                    break;
            }
            var aLink = document.createElement('a'),
                evt = document.createEvent("HTMLEvents"),
                blob = new Blob([content]);
            aLink.download = $('#node_text1').text() + '.' + type;
            aLink.href = URL.createObjectURL(blob);
            aLink.click();
            // alert("成功导出:"+aLink.download);
        }
    });
}

// 导出
$(document).on('click', '.diy export', function(event) {
    event.preventDefault();
    var type = $(this).data('type'),
            exportType;
    switch(type){
        case 'km':
            exportType = 'json';
            break;
        case 'md':
            exportType = 'markdown';
            break;
        default:
            exportType = type;
            break;
    }
    editor.minder.exportData(exportType).then(function(content){
        switch(exportType){
            case 'json':
                console.log($.parseJSON(content));
                break;
            default:
                console.log(content);
                break;
        }
        var aLink = document.createElement('a'),
                evt = document.createEvent("HTMLEvents"),
                blob = new Blob([content]);

        evt.initEvent("click", false, false);
        aLink.download = $('#node_text1').text()+'.'+type;
        aLink.href = URL.createObjectURL(blob);
        aLink.dispatchEvent(evt);
    });
});
