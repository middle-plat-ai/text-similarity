# -*- coding:utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn import metrics
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pickle
import sys

sys.path.append('model/')
from myBilstm import *
from create_dataset import *
import pandas as pd
import time


##
main_path = os.path.abspath('.')
data_path = main_path + '/data/'

try:
    dataset = pd.read_csv(data_path + 'dataset.csv', encoding='utf8')
    labels = dataset['label'].values
    features = np.loadtxt(data_path + "features.txt")
except:
    dataset = pd.read_table(data_path + 'dataset.txt', encoding='utf8')
    features, labels, data = create_features(dataset)
    features, embedding_word_matrix, nb_words, embedding_char_matrix, nb_chars = dl_features(data, features)



k = 5

kf = KFold(n_splits=k, random_state=2018 + 10)

params = {
    'learning_rate': 0.05,
    'max_depth': 7,
    'lambda_l1': 10,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': .7,
    'is_training_metric': False,
    'seed': 2018,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'scale_pos_weight': 1.5
}


def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = eval_gini(y, preds) / eval_gini(y, y)
    return 'gini', score, True


##
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

STAMP = 'lstm_%d_%d_%.2f_%.2f' % (128, 128, 0.2, 0.2)

checkpoint_dir = 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + STAMP + '.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

#preds = np.zeros((features.shape[0], 6))
ls = []
ratio = 0.5
for i, (train_index, test_index) in enumerate(kf.split(features)):
    samples = len(test_index)
    indices = np.arange(samples)
    np.random.shuffle(indices)
    num_val = int(samples * ratio)
    # Create data for this fold

    labels_train, label_test = labels[train_index], labels[test_index]
    labels_val, labels_test = label_test[:num_val], label_test[num_val:]
    features_train, feature_test = features[train_index], features[test_index]
    features_val, features_test = feature_test[:num_val], feature_test[num_val:]

    nlp_train = np.concatenate((features_train[:, :20], features_train[:, 80:630]), axis=1)
    nlp_val = np.concatenate((features_val[:, :20], features_val[:, 80:630]), axis=1)
    nlp_test = np.concatenate((feature_test[:, :20], feature_test[:, 80:630]), axis=1)

    statics_train = np.concatenate((features_train[:, 20:80], features_train[:, 630:1130]), axis=1)
    statics_val = np.concatenate((features_val[:, 20:80], features_val[:, 630:1130]), axis=1)
    statics_test = np.concatenate((feature_test[:, 20:80], feature_test[:, 630:1130]), axis=1)

    train_word_left = features_train[:, 1130:1204]
    train_word_right = features_train[:, 1204:1278]
    val_word_left = features_val[:, 1130:1204]
    val_word_right = features_val[:, 1204:1278]
    test_words_left = feature_test[:, 1130:1204]
    test_words_right = feature_test[:, 1204:1278]

    train_char_left = features_train[:, 1278:1434]
    train_char_right = features_train[:, 1434:]
    val_char_left = features_val[:, 1278:1434]
    val_char_right = features_val[:, 1434:]
    test_chars_left = feature_test[:, 1278:1434]
    test_chars_right = feature_test[:, 1434:]
    print(train_char_left.shape)
    print(train_char_right.shape)

    ###lgb模型
    print("第%d次交叉验证" % (i + 1))
    atec_model = lgb.train(
        params,
        lgb.Dataset(features_train[:, :1130], label=labels_train),
        3000,
        lgb.Dataset(features_val[:, :1130], label=labels_val),
        verbose_eval=50,
        feval=gini_lgb,
        early_stopping_rounds=1000,
        categorical_feature=[4, 5]
    )
    nlp_model = lgb.train(
        params,
        lgb.Dataset(nlp_train, label=labels_train),
        3000,
        lgb.Dataset(nlp_val, label=labels_val),
        verbose_eval=50,
        feval=gini_lgb,
        early_stopping_rounds=1000,
        categorical_feature=[4, 5]
    )
    statics_model = lgb.train(
        params,
        lgb.Dataset(statics_train, label=labels_train),
        3000,
        lgb.Dataset(statics_val, label=labels_val),
        verbose_eval=50,
        feval=gini_lgb,
        early_stopping_rounds=1000
    )
    print("Best iteration of atec_model = ", atec_model.best_iteration)
    # make prediction
    preds_all = atec_model.predict(feature_test[:, :1130], num_iteration=atec_model.best_iteration).reshape(-1, 1)
    print('log_loss', metrics.log_loss(label_test.astype(np.float64), preds_all))

    print("Best iteration of nlp_model = ", nlp_model.best_iteration)
    # make prediction
    preds_nlp = nlp_model.predict(nlp_test, num_iteration=nlp_model.best_iteration).reshape(-1, 1)
    print('log_loss', metrics.log_loss(label_test.astype(np.float64), preds_nlp))

    print("Best iteration of statics_model = ", statics_model.best_iteration)
    # make prediction
    preds_statics = statics_model.predict(statics_test, num_iteration=statics_model.best_iteration).reshape(-1, 1)
    print('log_loss', metrics.log_loss(label_test.astype(np.float64), preds_statics))

    # 深度学习模型
    model_word = my_bilstm(nb_words, embedding_word_matrix, MAX_SEQUENCE_LENGTH_1, 2, 128, 128)
    model_word.fit([train_word_left, train_word_right], labels_train, \
                   validation_data=([val_word_left, val_word_right], labels_val), nb_epoch=20, \
                   batch_size=256, shuffle=True, \
                   callbacks=[early_stopping, model_checkpoint, tensorboard])

    pred_word = model_word.predict([test_words_left, test_words_right])

    model_char = my_bilstm(nb_words, embedding_word_matrix, MAX_SEQUENCE_LENGTH_2, 2, 128, 128)
    model_char.fit([train_char_left, train_char_right], labels_train, \
                   validation_data=([val_char_left, val_char_right], labels_val), nb_epoch=20, \
                   batch_size=256, shuffle=True, \
                   callbacks=[early_stopping, model_checkpoint, tensorboard])

    pred_char = model_char.predict([test_chars_left, test_chars_right])

    model = my_bilstm_wc(nb_words, nb_chars, embedding_word_matrix, embedding_char_matrix, MAX_SEQUENCE_LENGTH_1,
                         MAX_SEQUENCE_LENGTH_2, 2, 160, 160)
    model.fit([train_word_left, train_char_left, train_word_right, train_char_right], labels_train,
              validation_data=([val_word_left, val_char_left, val_word_right, val_char_right], labels_val), nb_epoch=20,
              batch_size=256, shuffle=True, callbacks=[early_stopping, model_checkpoint, tensorboard])

    pred_wc = model.predict([test_words_left, test_chars_left, test_words_right, test_chars_right])

    pred = np.concatenate((preds_all, preds_nlp, preds_statics, pred_word, pred_char, pred_wc))

    # 保存模型
    with open(main_path + '/models/lgb_1130_' + str(i) + '.model', 'wb') as f:
        pickle.dump(atec_model, f)

    with open(main_path + '/models/lgb_nlp_570_' + str(i) + '.model', 'wb') as f:
        pickle.dump(atec_model, f)

    with open(main_path + '/models/lgb_statics_560_' + str(i) + '.model', 'wb') as f:
        pickle.dump(atec_model, f)

    json_string = model_word.to_json()  # json_string = model.get_config()
    open(main_path + '/models/word_model_architecture_' + str(i) + '.json', 'w').write(json_string)
    model_word.save_weights(main_path + '/models/word_model_weights_' + str(i) + '.h5')

    json_string = model_char.to_json()  # json_string = model.get_config()
    open(main_path + '/models/char_model_architecture_' + str(i) + '.json', 'w').write(json_string)
    model_char.save_weights(main_path + '/models/char_model_weights_' + str(i) + '.h5')

    json_string = model.to_json()  # json_string = model.get_config()
    open(main_path + '/models/model_architecture_' + str(i) + '.json', 'w').write(json_string)
    model.save_weights(main_path + '/models/model_weights_' + str(i) + '.h5')

    ls.append(pred)
preds = ls[0]
for i in range(1, len(ls)):
    preds = np.concatenate((preds, pred))

# 分割数据集
from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(preds, labels, test_size=0.2)

lgb_merge = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=120, max_depth=5, learning_rate=0.05,
                               n_estimators=2000, max_bin=100, subsample_for_bin=100000,
                               min_child_samples=10, subsample=0.8, colsample_bytree=0.8, seed=2018)

lgb_merge.fit(features_train, target_train)

# 输出测试的logloss
predicts = lgb_merge.predict_proba(features_test)
print('log_loss', metrics.log_loss(target_test.astype(np.float64), predicts))

# 保存和载入模型
import pickle

with open(main_path + '/models/lgb_merge_6.model', 'wb') as f:
    pickle.dump(lgb_merge, f)
print("模型训练完毕")