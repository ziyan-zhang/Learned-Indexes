# -*- coding: utf-8 -*-
# @File    : timestamp2ints.py
# @Date    : 2021-08-02 23:34
# @Author  : 张子岩
import pandas as pd
import numpy as np
from tqdm import tqdm
BLOCK_SIZE = 1000

data = pd.read_csv('weblog.csv', header=0)
def monthEng2int(s):
    i = 0
    months_eng = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    months_int = [11, 12, 1, 2, 3]
    while i < 5:
        if s == months_eng[i]:
            return months_int[i]
        i += 1
    raise ValueError

def timestamp2intList(s):
    l = np.array([])
    a1 = s[1:3]
    a2 = s[4:7]
    a3 = s[8:12]
    a4 = s[13:15]
    a5 = s[16:18]
    a6 = s[19:21]
    a = np.array([[int(s[1:3]), monthEng2int(s[4:7]), int(s[8:12]), int(s[13:15]), int(s[16:18]), int(s[19:21])]])
    l = np.append(l, a)
    return l

train_set_x = []
train_set_y = []
print("正在转换...")
for i in tqdm(range(data.shape[0])):
    ip = data.iloc[i, 0]
    timeStamp = data.iloc[i, 1]
    if len(ip) == 10 and len(timeStamp) == 21:
        train_set_x.append(timestamp2intList(timeStamp))
        train_set_y.append(int(i * data.shape[0]/BLOCK_SIZE))  # test_y即是位置, 也即

print("正在保存...")
dataset = pd.DataFrame.from_dict({'timeList': train_set_x, 'value': train_set_y})
dataset_path = 'weblog_new.csv'
dataset.to_csv(dataset_path, index=False)
