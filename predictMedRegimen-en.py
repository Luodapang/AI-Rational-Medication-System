import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
# from sklearn.model_selection import train_test_split
from model.data_process_ZM import process_predict_2cls_treatment
import math
import os
import json


"""  二分类：并可视化所有预测  """

PHY = True  # False
DDI = True
# file_path = "./data/prescription/prescription_2cls_all1.txt"  # 处方全部合理
# file_path = "./data/drugEco/data/prescription/标注/prescription_2cls.txt"  # 处方一半合理，一半不合理
# MODEL_PATH = f"./data/drugEco/transformer_Top3_treatment/final_model_DDI_{DDI}.model"

file_path = f"./data/drugEco/data/prescription/标注/prescription_2cls.txt"  # 处方一半合理，一半不合理
MODEL_PATH = f"./data/drugEco/transformer_Top3_treatment/final_model_DDI_{DDI}.model"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if os.name == 'nt':
#     model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# else:
#     model = torch.load(MODEL_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    model = torch.load(MODEL_PATH)
else:
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

Tx = 10
Ty = 12




# 数据划分一
# 划分训练集，测试集
# trainX, testX, trainY, testY = train_test_split(X, Y1, test_size=0.2, random_state=128)
# trainX, testX, true_train, true_test = train_test_split(X, Y2, test_size=0.2, random_state=128)
# # 保存测试集的诊断信息用以后续可视化系统
# patient_train, patient_test, phy_train, phy_test = train_test_split(patient, phy, test_size=0.2, random_state=128)

# phy_train, trainX, trainY, true_train
# phy_test, testX, testY, true_test

# 生成数据mask
def subsequent_mask(size):
    """
    mask后续的位置，返回[size, size]尺寸下三角Tensor
    对角线及其左下角全是1，右上角全是0
    """
    attn_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subseq_mask) == 0


# 构造数据
class Batch(object):
    "定义一个训练时需要的批次数据对象，封装了用于训练的src和tgt句子，以及mask"

    def __init__(self, src, trg=None, pad=0, phy_flag=False, ddi_flag=False):
        self.src = src  # B 个序列[1,5,3, 0]
        self.src_mask = (src != pad).unsqueeze(-2)  # [[1,1,1,0]]
        self.phy_flag = phy_flag
        # 如果没有生理信息，那么在src就只有Tx+2长度了，否则有Tx+4长度
        # 生理信息和ddi信息最终都会成为一个[nsamples,1,emb_dim]的向量
        # 因此在进行mask时，要考虑mask的长度
        # print(self.src_mask.size()) # torch.Size([2006, 1, 12]) ,都是bool格式
        if self.phy_flag:
            if not ddi_flag:
                self.src_mask = self.src_mask[:, :, 1:]
        else:
            if ddi_flag:
                ddi_mask = self.src_mask[:, :, :1]
                # print("ddi mask: ", ddi_mask.size()) # torch.Size([2006, 1, 1])
                self.src_mask = torch.cat([ddi_mask, self.src_mask], 2)
                # print("ddi mask after: ", self.src_mask.size()) # torch.Size([2006, 1, 13])

        if trg is not None:
            self.trg = trg[:, :-1]  #
            self.trg_y = trg[:, 1:]  # 后挪一个位置开始
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(data_src, data_tgt, phy_flag, ddi_flag):
    "Generate random data for a src-tgt copy task."
    data_src = torch.from_numpy(data_src)
    data_tgt = torch.from_numpy(data_tgt)
    src = Variable(data_src, requires_grad=False)
    tgt = Variable(data_tgt, requires_grad=False)
    yield Batch(src, tgt, 0, phy_flag, ddi_flag)



def greedy_decode_top1(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        # print("prob", prob)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    ys_top3 = [[1]]
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        # print("prob", prob)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        next_word_top3  = np.argsort(prob.detach().cpu().numpy()[0], axis=-1)[-3:]  # 三个备选
        ys_top3.append(list(next_word_top3))
        # pred_viz = [target_token_dict_inv[word] for word in pred_word_set]
    return ys, ys_top3




def eval_top3(src, src_mask, target_tokens, source_token_dict, target_token_dict):
    source_token_dict_inv = {v: k for k, v in source_token_dict.items()}
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
    # output.shape:  (Ty, drug_number)

    src_viz = [source_token_dict_inv[word] for word in src.cpu().numpy()[0]]
    print(f"0 - 可视化输入序列：{src_viz}")

    pred_words = []
    preds_viz = []
    # 可视化输出的序列
    print(f"1 - 可视化输出序列：{target_tokens[0]}")

    src_idx = src[0:1].to(device)
    # src_mask_idx = (src_idx != source_token_dict["<PAD>"]).unsqueeze(-2).to(device)
    src_mask_idx = src_mask[0]
    out, out_top3 = greedy_decode(model, src_idx, src_mask_idx,
                        max_len=Ty + 2, start_symbol=source_token_dict["<START>"])

    # 一个备选元素可视化
    drug_predict = []
    pred_words = []
    for j in range(1, out.size(1)):
        key = out[0][j].item()
        drug = target_token_dict_inv[key]
        if drug == "<END>": break
        drug_predict.append(drug)
        pred_words.append(key)
    print(f"3 - 一个备选元素可视化预测序列：{drug_predict}")



    # 三个备选元素可视化
    drug_predict = []
    pred_words = []
    for j in range(1, len(out_top3)):
        keys = out_top3[j]
        if 2 in keys:
            break
        drugs = []
        ids = []
        for key in keys:
            drug = target_token_dict_inv[key]
            drugs.append(drug)
            ids.append(key)
        drug_predict.append(drugs)
        pred_words.append(ids)
    print(f"4 - 三个备选元素可视化预测序列：{drug_predict}")

    return drug_predict



def eval_predict(data_iter, target_tokens, device, source_token_dict, target_token_dict):
    # 只有1个batch
    for i, batch in enumerate(data_iter):
        src = batch.src.to(device)
        src_mask = batch.src_mask.to(device)
        drug_predict = eval_top3(src, src_mask, target_tokens, source_token_dict, target_token_dict)
    return drug_predict


def predictMedRegimen(data):
    X, target_tokens, phy, ddi, patient, source_token_dict, source_token_dict_inv, target_token_dict, target_token_dict_inv = process_predict_2cls_treatment(data, Tx, Ty)

    # print("-------------原测试集-------------")
    drug_predict = eval_predict(data_gen(data_src=np.concatenate((phy, X), axis=1), data_tgt=X, phy_flag=PHY, ddi_flag=DDI),
                 target_tokens, device=device, source_token_dict=source_token_dict, target_token_dict=target_token_dict)
    return drug_predict


if __name__=="__main__":
    data = [4, "女", 98, 14, "癫痫,急性上呼吸道感染", "注射用头孢曲松钠(罗氏芬)_1天,0.9%氯化钠注射液(软袋)_1天"]
    data = ['4', '男', '106', '17.5', '急性支气管炎,支气管炎,急性上呼吸道感染', '阿莫西林克拉维酸钾干混悬剂(奥先)_3天,吸入用复方异丙托溴铵溶液(可必特)_1天,吸入用布地奈德混悬液(普米克令舒)_1天']
    predictMedRegimen(data)