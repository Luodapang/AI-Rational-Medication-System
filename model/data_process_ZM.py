from collections import defaultdict
import pandas as pd
import numpy as np
import dill
import os
import json

token_dict_file = f"./data/drugEco/data/dict/"
ddi_file = f"./data/drugEco/data/ZM_ddi_A_final_treatment.pkl"
sex_dict = {'女': 0, '男': 1}

# Generate dictionaries
def build_token_dict(token_list):
    token_dict = {'<PAD>': 0, '<START>': 1, '<END>': 2, }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

def process_predict_2cls_treatment(data, Tx, Ty):
    # 加载数据
    phy = []
    source_texts = []  # 目标文本列表
    target_texts = []  # 目标文本列表
    num = 1
    print(data)
    [age, sex, height, weight, source_text, target_text] = data
    phy.append([int(age) * 12, sex_dict[sex], int(float(weight) * 10)])
    source_texts.append(source_text.lower())  # 添加到输入文本序列
    target_texts.append(target_text.lower())  # 添加到目标文本序列
    source_tokens = [text.split(',') for text in source_texts]
    target_tokens = [text.split(',') for text in target_texts]

    with open(token_dict_file+"source_token_dict_treatment.json", "r", encoding="utf-8") as reader:
        content = reader.read()
        source_token_dict = json.loads(content)
    with open(token_dict_file+"target_token_dict_treatment.json", "r", encoding="utf-8") as reader:
        content = reader.read()
        target_token_dict = json.loads(content)


    source_token_dict_inv = {v: k for k, v in source_token_dict.items()}
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    meds = list(target_token_dict.keys())[3:]
    # 生成DDI矩阵

    # build_ddiMatrix(meds, target_token_dict)
    ddi = dill.load(open(ddi_file, "rb"))
    # print("ddi matrix: ",ddi.shape)
    # ddi = np.zeros((len(meds) + 3, len(meds) + 3))
    # dill.dump(ddi, open("data/ZM_ddi_A_final.pkl", 'wb'))

    '''
    2.4 - 数据转换
    在构造完诊断与药品映射表的基础上，我们此时将原始文本数据转化为数字编码。
    '''
    # Cutting And Add special tokens
    encode_tokens = [['<START>'] + tokens[:min(len(tokens), Tx)] + ['<END>'] for tokens in source_tokens]

    # 不足的进行补充
    source_max_len = max(map(len, encode_tokens))  # 求出source中的最长长度
    encode_tokens = [tokens + ['<PAD>'] * (Tx + 2 - len(tokens)) for tokens in encode_tokens]

    # 将句子进行编码，用数字表示
    encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]

    # 去除只有一个药品的处方
    t1 = []
    t4 = []
    t5 = []
    t1.append(encode_input[0])
    # t4.append(phy[i][2:])
    t4.append(phy[0])
    t5.append(phy[0])
    X = np.array(t1)
    phy = np.array(t4)
    patient = np.array(t5)
    # print(patient[:3])
    # print(patient.shape) # (924, 6)
    # print(phy[:3])
    # print(phy.shape) # (924, 4)
    return X.astype(np.int64), target_tokens, phy.astype(np.int64), ddi.astype(np.int64), patient.astype(np.int64), source_token_dict, source_token_dict_inv, target_token_dict, target_token_dict_inv
