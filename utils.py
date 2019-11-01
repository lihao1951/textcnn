#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : utils
Describe: 
    
Author : LH
Date : 2019/9/12
"""
import jieba
import jieba.analyse
import re, os

jieba.load_userdict(os.path.abspath('./config/newdict.txt'))


def remove_illegal_mark(content):
    """
    去除特殊标记
    :param content:
    :return:
    """
    pattern = re.compile(r'<[^>]+>|{.*?}|【.*?】|www.*?[conmf]{2,3}|http[s]{0,1}', re.S)
    content = content.strip().replace('\r', '').replace('\n', '')
    result = pattern.sub('', content)
    return result


def extract_keyword(text, topk, allowPos=('ns', 'n', 'vn', 'v', 'nr'), vocab=None):
    text = remove_illegal_mark(text)
    words = jieba.analyse.extract_tags(sentence=text, topK=topk, allowPOS=allowPos)
    wordlist = []
    for word in words:
        if len(word) < 2: continue
        if vocab:
            try:
                wordlist.append(vocab[word])
            except:
                wordlist.append(vocab['<PAD>'])
        else:
            wordlist.append(word)
    while len(wordlist) < topk and vocab:
        wordlist.append(vocab['<PAD>'])
    return wordlist[:topk]


def segmentation(text, topk, vocab=None):
    """
    返回分词后的结果，若vocab不为空，则返回词典中的ID
    :param text: 文本
    :param vocab: 给定词典
    :return: 返回词的列表
    """
    words = jieba.cut(text)
    wordlist = []
    for word in words:
        if len(word) < 2: continue
        if vocab:
            try:
                wordlist.append(vocab[word])
            except:
                wordlist.append(vocab['<PAD>'])
        else:
            wordlist.append(word)
    while len(wordlist) < topk and vocab:
        wordlist.append(vocab['<PAD>'])

    return wordlist[:topk]


def vocab():
    """
    词典的字典
    :return:
    """
    vocab = {}
    vocab_path = os.path.abspath('./config/vocab')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for ix, keyword in enumerate(f.readlines()):
            vocab[keyword.strip()] = ix
    return vocab


def process_data(topK=100, thresh=0.1):
    import random
    vocab_dict = vocab()
    w_train = open('yq_train.txt', 'w')
    w_valid = open('yq_valid.txt', 'w')
    w_test = open('yq_test.txt', 'w')
    with open('yq.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            data = line.split('_!_')
            if len(data) < 4:
                print('[data.size < 4]:  ', line)
                continue
            abstr = data[4]
            label = str(data[1])
            t = random.random()
            wordids = [str(id) for id in extract_keyword(abstr, topK, vocab=vocab_dict)]
            if t < thresh:
                w_test.write(label + ' ' + ' '.join(wordids) + '\n')
            elif t < thresh * 2:
                w_valid.write(label + ' ' + ' '.join(wordids) + '\n')
            else:
                w_train.write(label + ' ' + ' '.join(wordids) + '\n')

    w_train.close()
    w_test.close()
    w_valid.close()
