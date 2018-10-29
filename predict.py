# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pickle
from features_dataframe import *
from keras.models import model_from_json

main_path = os.path.abspath('.')

def load_dl_model(json_file_path, weights_path):
    loaded_model_json = open(json_file_path, 'r').read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model


text = '我要取消网商贷\t网商贷也是阿里巴巴的吗'
features = dl_features(create_features(pd.DataFrame([text.split('\t')], columns=['q1', 'q2'])))
features_nlp = np.concatenate((features[:, :20], features[:, 80:630]), axis=1)
features_statics = np.concatenate((features[:, 20:80], features[:, 630:1130]), axis=1)
words_left = features[:, 1130:1204]
words_right = features[:, 1204:1278]
chars_left = features[:, 1278:1434]
chars_right = features[:, 1434:]


# 载入模型
k = 4
preds = np.zeros((1, 6))
for i in range(k):
    print("第%d个模型" % (i + 1))
    # 载入lgb模型
    atec_model = pickle.load(open(main_path + '/models/lgb_1130_' + str(i) + '.model', 'rb'))
    nlp_model = pickle.load(open(main_path + '/models/lgb_nlp_570_' + str(i) + '.model', 'rb'))
    statics_model = pickle.load(open(main_path + '/models/lgb_statics_560_' + str(i) + '.model', 'rb'))

    # 载入深度学习模型
    word_json_file_path = main_path + '/models/word_model_architecture_' + str(i) + '.json'
    word_weights_path = main_path + '/models/word_model_weights_' + str(i) + '.h5'
    word_model = load_dl_model(word_json_file_path, word_weights_path)

    char_json_file_path = main_path + '/models/char_model_architecture_' + str(i) + '.json'
    char_weights_path = main_path + '/models/char_model_weights_' + str(i) + '.h5'
    char_model = load_dl_model(char_json_file_path, char_weights_path)

    json_file_path = main_path + '/models/model_architecture_' + str(i) + '.json'
    weights_path = main_path + '/models/model_weights_' + str(i) + '.h5'
    model = load_dl_model(json_file_path, weights_path)

    preds_all = atec_model.predict(features[:, :1130], num_iteration=atec_model.best_iteration).reshape(-1, 1)
    preds_nlp = nlp_model.predict(features_nlp, num_iteration=nlp_model.best_iteration).reshape(-1, 1)
    preds_statics = statics_model.predict(features_statics, num_iteration=statics_model.best_iteration).reshape(-1, 1)
    pred_word = word_model.predict([words_left, words_right])
    pred_char = char_model.predict([chars_left, chars_right])
    pred_wc = model.predict([words_left, chars_left, words_right, chars_right])
    pred = np.concatenate((preds_all, preds_nlp, preds_statics, pred_word, pred_char, pred_wc), axis=1)
    print(pred.shape)
    preds += pred


preds_avg = preds/k
print(preds_avg.shape)

##stacking_model
lgb_merge_6 = pickle.load(open(main_path + '/models/lgb_merge_6.model', 'rb'))
preds_stacking = lgb_merge_6.predict(preds_avg)
print(preds_stacking)