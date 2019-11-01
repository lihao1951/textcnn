#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : crawl
Describe: 
    今日头条新闻分类数据爬取
Author : LH
Date : 2019/9/9
"""

import requests
import json
import time
import random
import os,sys

g_cnns = [
    [100, '故事', 'news_story'],
    [101, '文化', 'news_culture'],
    [102, '娱乐', 'news_entertainment'],
    [103, '体育', 'news_sports'],
    [104, '财经', 'news_finance'],
    [105, '时尚', 'news_fashion'],
    [106, '房产', 'news_house'],
    [107, '汽车', 'news_car'],
    [108, '教育', 'news_edu' ],
    [109, '科技', 'news_tech'],
    [110, '军事', 'news_military'],
    [111, '历史', 'news_history'],
    [112, '旅游', 'news_travel'],
    [113, '国际', 'news_world'],
    [114, '股票', 'stock'],
    [115, '农业', 'news_agriculture'],
    [116, '游戏', 'news_game'],
    [117, '搞笑', 'funny'],
    [118, '美食', 'news_food'],
    [119, '育儿', 'news_baby'],
    [120, '养生', 'news_regimen']
]

g_ua = 'Dalvik/1.6.0 (Linux; U; Android 4.4.4; MuMu Build/V417IR) NewsArticle/6.3.1 okhttp/3.7.0.2'


g_id_cache = {}
g_count = 0

def get_data(tup):
    global g_id_cache
    global g_count
    cid = tup[0]
    cname = tup[2]
    url = "http://it.snssdk.com/api/news/feed/v63/"

    t = int(time.time()/10000)
    t = random.randint(6*t, 10*t)
    querystring = {"category":cname,"concern_id":"6215497896830175745","refer":"1","count":"20",
                   "max_behot_time":t,"last_refresh_sub_entrance_interval":"1524907088","loc_mode":"5",
                   "tt_from":"pre_load_more","cp":"51a5ee4f38c50q1","plugin_enable":"0",
                   "iid":"31047425023","device_id":"51425358841","ac":"wifi","channel":"tengxun",
                   "aid":"13","app_name":"news_article","version_code":"631","version_name":"6.3.1",
                   "device_platform":"android",
                   "ab_version":"333116,297979,317498,336556,295827,325046,239097,324283,170988,335432,332098,"
                                "325198,336443,330632,297058,276203,286212,313219,328615,332041,329358,322321,"
                                "327537,335710,333883,335102,334828,328670,324007,317077,334305,280773,335671,"
                                "319960,333985,331719,336452,214069,31643,332881,333968,318434,207253,266310,"
                                "321519,247847,281298,328218,335998,325618,333327,336199,323429,287591,288418,"
                                "260650,326188,324614,335477,271178,326588,326524,326532",
                   "ab_client":"a1,c4,e1,f2,g2,f7","ab_feature":"94563,102749","abflag":"3",
                   "ssmix":"a","device_type":"MuMu","device_brand":"Android",
                   "language":"zh","os_api":"19","os_version":"4.4.4",
                   "uuid":"008796762094657","openudid":"b7215ea70ca32066",
                   "manifest_version_code":"631","resolution":"1280*720","dpi":"240",
                   "update_version_code":"6310","_rticket":"1524907088018","plugin":"256"}
    headers = {
        'cache-control': "no-cache",
        'postman-token': "26530547-e697-1e8b-fd82-7c6014b3ee86",
        'User-Agent': g_ua
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    jj = json.loads(response.text)
    with open('toutiao_cat_data.txt', 'a',encoding='utf-8') as fp:
        for item in jj['data']:
            item = item['content']
            item = item.replace('\"', '"')
            item = json.loads(item)
            kws = ''
            if item.__contains__('keywords'):
                kws = item['keywords']
            if item.__contains__('ad_id'):
                print('ad')
            elif not item.__contains__('item_id') or not item.__contains__('title'):
                print('bad')
            else:
                item_id = item['item_id']
                item_abstr = item['abstract'].replace('\n','').replace('\r','')
                # print(g_count, cid, cname, item['item_id'], item['title'], kws)
                if g_id_cache.__contains__(item_id):
                    print('dulp')
                else:
                    g_id_cache[item_id] = 1
                    line = u"{}_!_{}_!_{}_!_{}_!_{}".format(item['item_id'], cid, cname, item['title'], item_abstr)
                    line = line.replace('\n', '').replace('\r', '')
                    print(line)
                    line = line + '\n'
                    fp.write(line)
                    g_count += 1

def get_routine():
    global g_count
    if os.path.exists('./toutiao_cat_data.txt'):
        with open('toutiao_cat_data.txt', 'r',encoding='utf-8') as fp:
            ll = fp.readlines()
            g_count = len(ll)
            for l in ll:
                ww = l.split('_!_')
                item_id = int(ww[0])
                g_id_cache[item_id] = 1
            print('load cache done, ', g_count)

    while 1:
        time.sleep(30)
        for tp in g_cnns:
            get_data(tp)

get_routine()
