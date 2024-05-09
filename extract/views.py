from django.shortcuts import render
import sys
sys.path.append(r'..')
from BERT_BiLSTM_CRF_NER.extract_smart import read_docx, create_directory, extract_one_sample
import os
absp = os.path.abspath('.')
# print(absp)
from datetime import datetime
# Create your views here.

files = ""
def ihomepage(request):
    return render(request, 'ihomepage.html')
def ileft(request):
    field_dict = ""
    content_docx = ""
    drug_json = ""
    global files
    if request.method == 'POST':
        file = request.FILES.get('fileSelect')
        if file:
            files = file.name
            create_directory()
            destination = open(os.path.join(f"{absp}/data/instructions", files), 'wb+')  # 打开特定的文件进行二进制的写操作
            for chunk in file.chunks():  # 分块写入文件
                destination.write(chunk)
            destination.close()
            files = str(files)
            print(files)
        if files:
            path_instructions = f"{absp}/data/instructions"
            try:
                if files.find(" ") != -1:
                    src = f'{path_instructions}/{files}'
                    files = files.replace(" ", "")
                    dst = f'{path_instructions}/{files}'
                    os.rename(src, dst)  # 无法重命名？
                content_docx = read_docx(files)
            except:
                content_docx = "<center><h1>文件读取失败</h1></center>"
            if 'highlight' in request.POST:
                # start = datetime.now()
                try:
                    field_dict, drug_json = extract_one_sample(files)
                except Exception as e:
                    content_docx = str(e)
                drug_json = str(drug_json).replace("\'", '\\"')
                drug_json = drug_json.replace("None", "null")
                drug_json = drug_json.replace("\n", "\r\n")
                drug_json = '\"' + drug_json + '\"'

    return render(request, 'ileft.html', {'field_dict': field_dict, "content_docx": content_docx, "drug_json": drug_json})


def iright(request):
    return render(request, 'iright.html')

