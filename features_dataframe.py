# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
import pandas as pd
import time
import sys

sys.path.append('utils/')
sys.path.append('FeatureEngineer/')
from extractor_features import *
from My_doc2vec import Doc_vectors
from preprocess import *
from cut_words import *


# from text_solver import *
# from My_doc2vec import Doc_vectors
# from extractor_features import *
# from langconv import *

doc_vec = Doc_vectors()

main_path = os.path.abspath('.')
dict_path = main_path + '/dict/'

#停用词库
stopwords_path = dict_path + 'stop_word.txt'
stop_set = set([value.strip().replace('\n', '') for value in open(stopwords_path, encoding='utf8').readlines()])


#load tfidf model
with open(main_path + '/models/t.pkl', 'rb') as f:
    t = pickle.load(f)

with open(main_path + '/models/t_clean.pkl', 'rb') as f2:
    t_clean = pickle.load(f2)

with open(main_path + '/models/t_char.pkl', 'rb') as f3:
    t_char = pickle.load(f3)

with open(main_path + '/models/t_char_clean.pkl', 'rb') as f4:
    t_char_clean = pickle.load(f4)

with open(main_path + '/models/com_chi2.pkl', 'rb') as f5:
    com_chi2 = pickle.load(f5)
    
with open(main_path + '/models/lda.pkl', 'rb') as g1:
        lda = pickle.load(g1)

with open(main_path + '/models/lda_clean.pkl', 'rb') as g2:
    lda_clean = pickle.load(g2)

with open(main_path + '/models/model_dm_word.pkl', 'rb') as g3:
    model_dm_word = pickle.load(g3)

with open(main_path + '/models/model_dm_char.pkl', 'rb') as g4:
    model_dm_char = pickle.load(g4)

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

from pyhanlp import *

#分词不去停用词
def seg_cut_pos(text):
    return ' '.join([str(term) for term in HanLP.segment(text.strip())])

def seg_cut(text):
    return ' '.join([str(term).split('/')[0] for term in HanLP.segment(text.strip())])


#分词去停用词
def seg_cut_stopset(text):
    return ' '.join([str(term).split('/')[0] for term in HanLP.segment(text.strip()) \
                     if str(term).split('/')[0] not in stop_set])

def clean_poscut(text):
    text1, text2 = text.split('\t')
    words = set(text2.split())
    return ' '.join([value for value in text1.split() if value.split('/')[0] in words])

def create_features(dataset):
    #traditional to simple
    dataset['text1'] = dataset['q1'].str.strip().str.lower().map(Traditional2Simplified)
    dataset['text2'] = dataset['q2'].str.strip().str.lower().map(Traditional2Simplified)

    #
    f = open(dict_path + 'words_replace.txt', 'r', encoding='utf8')
    for line in f.readlines():
        value = line.strip().replace(r'\n', '').split(',')
        s1 = value[0]
        s2 = value[1]
        dataset['text1'] = dataset['text1'].str.replace(s1, s2)
        dataset['text2'] = dataset['text2'].str.replace(s1, s2)

    # 带词性的分词结果
    dataset['pos_cut1'] = dataset['text1'].map(seg_cut_pos)
    dataset['pos_cut2'] = dataset['text2'].map(seg_cut_pos)

    # 不带词性的分词结果
    dataset['cut1'] = dataset['text1'].map(seg_cut)
    dataset['cut2'] = dataset['text2'].map(seg_cut)
    dataset['stop_cut1'] = dataset['text1'].map(seg_cut_stopset)
    dataset['stop_cut2'] = dataset['text2'].map(seg_cut_stopset)
    # 对分词结果进行清洗
    dataset['cut_clean1'] = dataset['stop_cut1'].map(cleantext)
    dataset['cut_clean2'] = dataset['stop_cut2'].map(cleantext)

    temp_texts1 = dataset['pos_cut1'] + '\t' + dataset['cut_clean1']
    temp_texts2 = dataset['pos_cut2'] + '\t' + dataset['cut_clean2']

    dataset['pos_cut_clean1'] = temp_texts1.map(clean_poscut)
    dataset['pos_cut_clean2'] = temp_texts2.map(clean_poscut)

    # 词
    dataset['words_rmsame'] = dataset.apply(lambda row: rm_same(row['cut1'], row['cut2']), axis=1)
    dataset['words_clean_rmsame'] = dataset.apply(lambda row: rm_same(row['cut_clean1'], row['cut_clean2']), axis=1)
    dataset['words_clean_rmsame1'] = dataset['words_clean_rmsame'].map(lambda x: x.split('\t')[0])
    dataset['words_clean_rmsame2'] = dataset['words_clean_rmsame'].map(lambda x: x.split('\t')[1])

    dataset['words_pos_rmsame'] = dataset.apply(lambda row: rm_same(row['pos_cut1'], row['pos_cut2']), axis=1)
    dataset['words_pos_clean_rmsame'] = dataset.apply(lambda row: rm_same(row['pos_cut_clean1'], row['pos_cut_clean2']), axis=1)

    # 字符
    dataset['char1'] = dataset['text1'].map(get_char)
    dataset['char2'] = dataset['text2'].map(get_char)
    dataset['char_clean1'] = dataset['cut_clean1'].map(get_char).map(cleantext)
    dataset['char_clean2'] = dataset['cut_clean2'].map(get_char).map(cleantext)

    dataset['chars_rmsame'] = dataset.apply(lambda row: rm_same(row['char1'], row['char2']), axis=1)
    dataset['chars_clean_rmsame'] = dataset.apply(lambda row: rm_same(row['char_clean1'], row['char_clean2']), axis=1)
    dataset['chars_clean_rmsame1'] = dataset['chars_clean_rmsame'].map(lambda x: x.split('\t')[0])
    dataset['chars_clean_rmsame2'] = dataset['chars_clean_rmsame'].map(lambda x: x.split('\t')[1])

    dataset['jac_dis_word'] = dataset.apply(lambda row: jac_dis(row['cut1'], row['cut2']), axis=1)
    dataset['jac_dis_word_clean'] = dataset.apply(lambda row: jac_dis(row['cut_clean1'], row['cut_clean2']), axis=1)

    dataset['Levenshtein_word'] = dataset.apply(lambda row: Leven_ratio(row['cut1'], row['cut2']), axis=1)
    dataset['Leven_distince_word'] = dataset.apply(lambda row: Leven_dis(row['cut1'], row['cut2']), axis=1)
    dataset['Levenshtein_word_clean'] = dataset.apply(lambda row: Leven_ratio(row['cut_clean1'], row['cut_clean2']),
                                                      axis=1)
    dataset['Leven_distince_word_clean'] = dataset.apply(lambda row: Leven_dis(row['cut_clean1'], row['cut_clean2']),
                                                         axis=1)

    ##fuzz距离特征
    dataset['fuzz_ratio_word'] = dataset.apply(lambda row: fuzz_ratio(row['cut1'], row['cut2']), axis=1)
    dataset['fuzz_ratio_word_clean'] = dataset.apply(lambda row: fuzz_ratio(row['cut_clean1'], row['cut_clean2']),
                                                     axis=1)
    dataset['fpartial_ratio_word'] = dataset.apply(lambda row: fpartial_ratio(row['cut1'], row['cut2']), axis=1)
    dataset['fpartial_ratio_word_clean'] = dataset.apply(
        lambda row: fpartial_ratio(row['cut_clean1'], row['cut_clean2']), axis=1)
    dataset['ftoken_sort_ratio_word'] = dataset.apply(lambda row: ftoken_sort_ratio(row['cut1'], row['cut2']), axis=1)
    dataset['ftoken_sort_ratio_word_clean'] = dataset.apply(
        lambda row: ftoken_sort_ratio(row['cut_clean1'], row['cut_clean2']), axis=1)
    dataset['ftoken_set_ratio_word'] = dataset.apply(lambda row: ftoken_set_ratio(row['cut1'], row['cut2']), axis=1)
    dataset['ftoken_set_ratio_word_clean'] = dataset.apply(
        lambda row: ftoken_set_ratio(row['cut_clean1'], row['cut_clean2']), axis=1)

    # 句子长度(字符的个数，包括标点符号)
    dataset['q1_len'] = dataset['q1'].map(len)
    dataset['q2_len'] = dataset['q2'].map(len)
    dataset['text1_len'] = dataset['text1'].map(len)
    dataset['text2_len'] = dataset['text2'].map(len)
    # 词的个数
    dataset['cut1_len'] = dataset['cut1'].map(lambda x: len(x.split(' ')))
    dataset['cut2_len'] = dataset['cut2'].map(lambda x: len(x.split(' ')))
    dataset['stop_cut1_len'] = dataset['stop_cut1'].map(lambda x: len(x.split(' ')))
    dataset['stop_cut2_len'] = dataset['stop_cut2'].map(lambda x: len(x.split(' ')))
    dataset['cut_clean1_len'] = dataset['cut_clean1'].map(lambda x: len(x.split(' ')))
    dataset['cut_clean2_len'] = dataset['cut_clean2'].map(lambda x: len(x.split(' ')))
    dataset['words_rmsame_len1'] = dataset['words_rmsame'].map(lambda x: len((x.split('\t')[0]).split()))
    dataset['words_rmsame_len2'] = dataset['words_rmsame'].map(lambda x: len((x.split('\t')[1]).split()))
    dataset['words_clean_rmsame_len1'] = dataset['words_clean_rmsame'].map(lambda x: len((x.split('\t')[0]).split()))
    dataset['words_clean_rmsame_len2'] = dataset['words_clean_rmsame'].map(lambda x: len((x.split('\t')[1]).split()))
    dataset['common_size'] = dataset.apply(lambda row: len(common_words(row['cut1'], row['cut2'])), axis=1)
    dataset['common_size_clean'] = dataset.apply(lambda row: len(common_words(row['cut_clean1'], row['cut_clean2'])),
                                                 axis=1)
    dataset['common_rate1'] = dataset['common_size'] * 1.0 / dataset['cut1_len']
    dataset['common_rate2'] = dataset['common_size'] * 1.0 / dataset['cut2_len']
    dataset['common_clean_rate1'] = dataset['common_size_clean'] * 1.0 / dataset['cut_clean1_len']
    dataset['common_clean_rate2'] = dataset['common_size_clean'] * 1.0 / dataset['cut_clean2_len']

    dataset['words_rmsame_len1'] = dataset['words_rmsame'].map(lambda x: len((x.split('\t')[0]).split()))
    dataset['words_rmsame_len2'] = dataset['words_rmsame'].map(lambda x: len((x.split('\t')[1]).split()))
    dataset['words_clean_rmsame_len1'] = dataset['words_clean_rmsame'].map(lambda x: len((x.split('\t')[0]).split()))
    dataset['words_clean_rmsame_len2'] = dataset['words_clean_rmsame'].map(lambda x: len((x.split('\t')[1]).split()))

    #####字符特征
    # 字符的个数
    dataset['char_clean1_len'] = dataset['char_clean1'].map(lambda x: len(x.split(' ')))
    dataset['char_clean2_len'] = dataset['char_clean2'].map(lambda x: len(x.split(' ')))

    dataset['jac_dis_char'] = dataset.apply(lambda row: jac_dis(row['char1'], row['char2']), axis=1)
    dataset['jac_dis_char_clean'] = dataset.apply(lambda row: jac_dis(row['char_clean1'], row['char_clean2']), axis=1)

    dataset['Levenshtein_char'] = dataset.apply(lambda row: Leven_ratio(row['char1'], row['char2']), axis=1)
    dataset['Leven_distince_char'] = dataset.apply(lambda row: Leven_dis(row['char1'], row['char2']), axis=1)
    dataset['Levenshtein_char_clean'] = dataset.apply(lambda row: Leven_ratio(row['char_clean1'], row['char_clean2']),
                                                      axis=1)
    dataset['Leven_distince_char_clean'] = dataset.apply(lambda row: Leven_dis(row['char_clean1'], row['char_clean2']),
                                                         axis=1)

    ##fuzz距离特征
    dataset['fuzz_ratio_char'] = dataset.apply(lambda row: fuzz_ratio(row['char1'], row['char2']), axis=1)
    dataset['fuzz_ratio_char_clean'] = dataset.apply(lambda row: fuzz_ratio(row['char_clean1'], row['char_clean2']),
                                                     axis=1)
    dataset['fpartial_ratio_char'] = dataset.apply(lambda row: fpartial_ratio(row['char1'], row['char2']), axis=1)
    dataset['fpartial_ratio_char_clean'] = dataset.apply(
        lambda row: fpartial_ratio(row['char_clean1'], row['char_clean2']), axis=1)
    dataset['ftoken_sort_ratio_char'] = dataset.apply(lambda row: ftoken_sort_ratio(row['char1'], row['char2']), axis=1)
    dataset['ftoken_sort_ratio_char_clean'] = dataset.apply(
        lambda row: ftoken_sort_ratio(row['char_clean1'], row['char_clean2']), axis=1)
    dataset['ftoken_set_ratio_char'] = dataset.apply(lambda row: ftoken_set_ratio(row['char1'], row['char2']), axis=1)
    dataset['ftoken_set_ratio_char_clean'] = dataset.apply(
        lambda row: ftoken_set_ratio(row['char_clean1'], row['char_clean2']), axis=1)

    # 共有词特征
    dataset['common_char_size'] = dataset.apply(lambda row: len(common_words(row['char1'], row['char2'])), axis=1)
    dataset['common_char_size_clean'] = dataset.apply(
        lambda row: len(common_words(row['char_clean1'], row['char_clean2'])), axis=1)
    dataset['common_char_rate1'] = dataset['common_char_size'] * 1.0 / dataset['text1_len']
    dataset['common_char_rate2'] = dataset['common_char_size'] * 1.0 / dataset['text2_len']
    dataset['common_char_clean_rate1'] = dataset['common_char_size_clean'] * 1.0 / dataset['char_clean1_len']
    dataset['common_char_clean_rate2'] = dataset['common_char_size_clean'] * 1.0 / dataset['char_clean2_len']

    dataset['chars_rmsame_len1'] = dataset['chars_rmsame'].map(lambda x: len((x.split('\t')[0]).split()))
    dataset['chars_rmsame_len2'] = dataset['chars_rmsame'].map(lambda x: len((x.split('\t')[1]).split()))
    dataset['chars_clean_rmsame_len1'] = dataset['chars_clean_rmsame'].map(lambda x: len((x.split('\t')[0]).split()))
    dataset['chars_clean_rmsame_len2'] = dataset['chars_clean_rmsame'].map(lambda x: len((x.split('\t')[1]).split()))

    # 分别统计q1和q2中名词和动词的数量
    def v_size(text):
        if text != '':
            return len([value for value in text.split() if value.split('/')[1].find('v') >= 0])
        else:
            return 0

    def noun_size(text):
        if text != '':
            return len([value for value in text.split() if value.split('/')[1].find('n') >= 0])
        else:
            return 0

    dataset['v_clean_size1'] = dataset['words_pos_clean_rmsame'].map(lambda x: v_size(x.split('\t')[0]))
    dataset['v_clean_size2'] = dataset['words_pos_clean_rmsame'].map(lambda x: v_size(x.split('\t')[1]))
    dataset['v_clean_size_rate1'] = dataset['v_clean_size1'] * 1.0 / dataset['words_clean_rmsame_len1']
    dataset['v_clean_size_rate2'] = dataset['v_clean_size2'] * 1.0 / dataset['words_clean_rmsame_len2']
    dataset['noun_clean_size1'] = dataset['words_pos_clean_rmsame'].map(lambda x: noun_size(x.split('\t')[0]))
    dataset['noun_clean_size2'] = dataset['words_pos_clean_rmsame'].map(lambda x: noun_size(x.split('\t')[1]))
    dataset['noun_clean_size_rate1'] = dataset['noun_clean_size1'] * 1.0 / dataset['words_clean_rmsame_len1']
    dataset['noun_clean_size_rate2'] = dataset['noun_clean_size2'] * 1.0 / dataset['words_clean_rmsame_len2']

    dataset['v_clean_size_rate1'] = dataset['v_clean_size_rate1'].fillna(0)
    dataset['v_clean_size_rate2'] = dataset['v_clean_size_rate2'].fillna(0)
    dataset['noun_clean_size_rate1'] = dataset['v_clean_size_rate1'].fillna(0)
    dataset['noun_clean_size_rate2'] = dataset['v_clean_size_rate2'].fillna(0)

    # 关联上q1和q2的频次、句式和情感类型
    df_q = pd.read_table(main_path + '/features/question_freq.txt')
    dataset = dataset.merge(df_q, left_on='q1', right_on='q', how='left')
    dataset = dataset.merge(df_q, left_on='q2', right_on='q', how='left')
    dataset = dataset.drop(['q_x', 'q_y'], axis=1)

    # 判断q1与q2的句式和情感类型是否相同
    def type_same(text):
        t1, t2 = text.split('\t')
        if t1 == t2:
            return 1
        else:
            return 0

    dataset['times_avg'] = (dataset['times_x'] + dataset['times_y']) / 2.0
    dataset['times_dif'] = (dataset['times_x'] - dataset['times_y']).abs()
    dataset['types_xy'] = dataset['types_x'] + '\t' + dataset['types_y']
    dataset['sentiments_xy'] = dataset['sentiments_x'] + '\t' + dataset['sentiments_y']
    dataset['types_label'] = dataset['types_xy'].map(type_same)
    dataset['sentiments_label'] = dataset['sentiments_xy'].map(type_same)

    # 删掉无用的列
    dataset = dataset.drop(['sentiments_xy', 'types_xy', 'sentiments_y', 'sentiments_x', 'types_y', 'types_x'], axis=1)

    #tfidf相似性
    dataset['tfidf_similar'] = dataset.apply(lambda row: cos_similar(row['cut1'], row['cut2'], t), axis=1)
    dataset['tfidf_clean_similar'] = dataset.apply(lambda row: cos_similar(row['cut_clean1'], row['cut_clean2'], t_clean), axis=1)
    vector1 = t_clean.transform(dataset['cut_clean1'])
    vector2 = t_clean.transform(dataset['cut_clean2'])
    vec_new = abs(vector1 + vector2)
    features_chi2 = com_chi2.transform(vec_new)

    dataset['tfidf_char_similar'] = dataset.apply(lambda row: cos_similar(row['char1'], row['char2'], t_char), axis=1)
    dataset['tfidf_char_clean_similar'] = dataset.apply(
        lambda row: cos_similar(row['char_clean1'], row['char_clean2'], t_char_clean), axis=1)

    #主题模型
    lda_docs1 = lda_clean.transform(vector1)
    lda_docs2 = lda_clean.transform(vector2)
    lda_docs = lda_clean.transform(vec_new)

    # 把lda的结果拼接起来
    lda_merge = np.concatenate((lda_docs, lda_docs1, lda_docs2), axis=1)
    dataset['lda_similar'] = dataset.apply(lambda row: cos_similar_lda(row['cut1'], row['cut2'], t, lda), axis=1)
    dataset['lda_clean_similar'] = dataset.apply(
        lambda row: cos_similar_lda(row['cut_clean1'], row['cut_clean2'], t_clean, lda_clean), axis=1)

    #句向量
    doc2vec_word_1 = doc_vec.getVecs(model_dm_word, doc_vec.list2tag(dataset['cut1']))
    doc2vec_word_2 = doc_vec.getVecs(model_dm_word, doc_vec.list2tag(dataset['cut2']))


    doc2vec_char_1 = doc_vec.getVecs(model_dm_char, doc_vec.list2tag(dataset['char1']))
    doc2vec_char_2 = doc_vec.getVecs(model_dm_char, doc_vec.list2tag(dataset['char2']))
    doc2vec_merge = np.concatenate((doc2vec_word_1, doc2vec_word_2, doc2vec_char_1, doc2vec_char_2), axis=1)

    # 调整列的顺序
    values = ['lda_clean_similar', 'lda_similar', 'tfidf_char_clean_similar', 'tfidf_char_similar',
              'tfidf_clean_similar', 'tfidf_similar', 'noun_clean_size_rate2', 'noun_clean_size_rate1',
              'noun_clean_size2', 'noun_clean_size1', 'v_clean_size_rate2', 'v_clean_size_rate1', 'v_clean_size2',
              'v_clean_size1', 'sentiments_label', 'types_label', 'times_dif', 'times_avg', 'times_y', 'times_x']

    for value in values:
        df = dataset[value]
        dataset.drop(columns=[value], axis=1, inplace=True)
        dataset.insert(28, value, df)

    #把所有的特征拼接起来
    lda_merge_df = pd.DataFrame(lda_merge, columns=list(range(150)))
    doc2vec_merge_df = pd.DataFrame(doc2vec_merge, columns=list(range(150, 550)))
    df_chi2 = pd.DataFrame(features_chi2.toarray(), columns=list(range(550, 1050)))
    features = pd.concat([dataset, lda_merge_df, doc2vec_merge_df, df_chi2], axis=1)
    return features

####深度学习特征
from keras.preprocessing.sequence import pad_sequences

#模型参数
MAX_SEQUENCE_LENGTH_1 = 37
MAX_SEQUENCE_LENGTH_2 = 78
EMBEDDING_DIM = 300
MAX_NB_words = 200000
MAX_NB_chars = 200000

##载如模型
with open(main_path + '/models/word_tokenizer.model', 'rb') as f1:
    word_tokenizer = pickle.load(f1)

with open(main_path + '/models/char_tokenizer.model', 'rb') as f2:
    char_tokenizer = pickle.load(f2)


def dl_features(features):
    word_corpus = pd.concat([features['cut_clean1'], features['cut_clean2']], axis=0)
    char_corpus = pd.concat([features['char_clean1'], features['char_clean2']], axis=0)

    word_vocab = word_tokenizer.word_index
    char_vocab = char_tokenizer.word_index

    # 将每个词用词典中的数值代替
    sequences_word_left1 = features['cut_clean1'].apply(lambda x: text2int(x, word_vocab))
    sequences_word_right1 = features['cut_clean2'].apply(lambda x: text2int(x, word_vocab))
    sequences_word_left2 = features['words_clean_rmsame1'].apply(lambda x: text2int(x, word_vocab))
    sequences_word_right2 = features['words_clean_rmsame2'].apply(lambda x: text2int(x, word_vocab))

    # 将每个字用词典中的数值代替
    sequences_char_left1 = features['char_clean1'].apply(lambda x: text2int(x, char_vocab))
    sequences_char_right1 = features['char_clean2'].apply(lambda x: text2int(x, char_vocab))
    sequences_char_left2 = features['chars_clean_rmsame1'].apply(lambda x: text2int(x, char_vocab))
    sequences_char_right2 = features['chars_clean_rmsame2'].apply(lambda x: text2int(x, char_vocab))

    # 序列模式
    word_left1 = pad_sequences(sequences_word_left1, maxlen=MAX_SEQUENCE_LENGTH_1, \
                               padding='pre', truncating='post')
    word_right1 = pad_sequences(sequences_word_right1, maxlen=MAX_SEQUENCE_LENGTH_1, \
                                truncating='post')
    word_left2 = pad_sequences(sequences_word_left2, maxlen=MAX_SEQUENCE_LENGTH_1, \
                               padding='pre', truncating='post')
    word_right2 = pad_sequences(sequences_word_right2, maxlen=MAX_SEQUENCE_LENGTH_1, \
                                truncating='post')
    word_left = np.concatenate((word_left1, word_left2), axis=1)
    word_right = np.concatenate((word_right1, word_right2), axis=1)

    char_left1 = pad_sequences(sequences_char_left1, maxlen=MAX_SEQUENCE_LENGTH_2, \
                               padding='pre', truncating='post')
    char_right1 = pad_sequences(sequences_char_right1, maxlen=MAX_SEQUENCE_LENGTH_2, \
                                truncating='post')
    char_left2 = pad_sequences(sequences_char_left2, maxlen=MAX_SEQUENCE_LENGTH_2, \
                               padding='pre', truncating='post')
    char_right2 = pad_sequences(sequences_char_right2, maxlen=MAX_SEQUENCE_LENGTH_2, \
                                truncating='post')
    char_left = np.concatenate((char_left1, char_left2), axis=1)
    char_right = np.concatenate((char_right1, char_right2), axis=1)

    features = np.concatenate((features.iloc[:, 28:].values, word_left, word_right, char_left, char_right), axis=1)
    return features



if __name__ == '__main__':
    start = time.clock()
    text = '我要取消网商贷\t网商贷也是阿里巴巴的吗'
    features = dl_features(create_features(pd.DataFrame([text.split('\t')], columns=['q1', 'q2'])))
    print(features.shape)
    end = time.clock()
    print(end-start)