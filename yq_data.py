#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : yq_data
Describe: 
    
Author : LH
Date : 2019/9/17
"""
import requests
import json
from utils import remove_illegal_mark

label_names = {0: '体育', 1: '母婴', 2: '教育', 3: '科技', 4: '政务', 5: '文化', 6: '美容', 7: '国际', 8: '军事',
               9: '民生', 10: '时尚', 11: '娱乐', 12: '游戏', 13: '财经', 14: '股票', 15: '历史', 16: '美食',
               17: '房产', 18: '汽车'}


def ask_data(url, data):
    data = json.dumps(data)
    resp = requests.post(url, data)
    hits = json.loads(resp.text)
    return hits['hits']['hits']


def save_yq_data():
    url = 'http://10.1.101.55:9200/base_wde_app/app_type/_search?pretty'
    w = open('yq.txt', 'w', encoding='utf-8')
    id_map = {}
    for k, v in label_names.items():
        for f in range(0, 5000, 500):
            param = {'query': {'match': {'i_bn': v}}, '_source': ['title', 'cont', 'i_bn'], 'size': 500, 'from': f}
            print('query param:', param)
            resp = ask_data(url, param)
            for d in resp:
                id = d['_id']
                if id in id_map.keys():
                    print('{} is existed'.format(id))
                else:
                    source = d['_source']
                    ibn = source['i_bn']
                    title = remove_illegal_mark(source['title'])
                    cont = remove_illegal_mark(source['cont'])
                    if (len(cont) > 80) and (v in ibn):
                        id_map[id] = 1
                        newline = '{}_!_{}_!_{}_!_{}_!_{}\n'.format(id, k, v, title, cont)
                        w.write(newline)
    w.close()
