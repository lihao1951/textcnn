#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/22
"""
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template,jsonify
from classification import TextCNN
import utils
import json

label_names = {0: '体育', 1: '母婴', 2: '教育', 3: '科技', 4: '政务', 5: '文化', 6: '美容', 7: '国际', 8: '军事',
               9: '民生', 10: '时尚', 11: '娱乐', 12: '游戏', 13: '时事', 14: '股票', 15: '历史', 16: '美食',
               17: '房产', 18: '汽车'}
app = Flask(__name__)
topK = 100
MODEL_NAME = './model/yq/'
vocab = utils.vocab()
textcnn = TextCNN(sequence_length=100, num_classes=len(label_names), vocab_size=len(vocab), embedding_size=100,
                  filter_sizes=[3, 5, 7, 9], num_filters=64, trainable=False,pool_k=1)
textcnn.load_model(MODEL_NAME)


@app.route('/')
def demo():
    return render_template('index.html')


@app.route('/golaxy/classify/textcnn',methods=['POST'])
def text_summary():
    data = request.get_data().decode('utf-8')
    data = json.loads(data)
    contid = []
    for d in data:
        cont = d['cont']
        contid.append(utils.extract_keyword(cont, topK, vocab=vocab))
    features = np.array(contid, dtype=np.int)
    predict = textcnn.inference(features)
    predict = [str(ix) for ix in predict]
    result = ''
    for p in predict:
        result += ' '+p
    # predict = [label_names[ix] for ix in predict]
    return jsonify(result)

@app.route('/predict/yqcnn', methods=['GET', 'POST'])
def predict_yqcnn():
    cont = request.form.get('cont')
    features = np.array([utils.extract_keyword(cont, topK, vocab=vocab)], dtype=np.int)
    predict = textcnn.inference(features)
    ix = int(predict[0])
    result = label_names[ix]
    return render_template('predict.html', predict=result)


if __name__ == '__main__':
    app.run(debug=False, port=9999, host='0.0.0.0')
