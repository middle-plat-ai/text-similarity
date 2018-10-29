# -*- coding: utf-8 -*-

from numpy import *
from fuzzywuzzy import fuzz
import numpy as np

#计算词的个数
def cal_length(text):
    return len(text.split())


#字符
def get_char(text):
    if text != '':
        return ' '.join([value for value in text.replace(' ', '')])


#drop common words/chars
def rm_same(text1, text2):
    ls_1, ls_2 = text1.split(), text2.split()
    common = set(ls_1) & set(ls_2)
    text1_new = ' '.join([value for value in ls_1 if value not in common])
    text2_new = ' '.join([value for value in ls_2 if value not in common])
    return text1_new+'\t'+text2_new


#q1和q2中共同词的个数(分去停用词和不去停用词)在每句话中的比例，去掉共同词之后的长度
def common_words(text1, text2):
    #text1, text2 = text.split('\t')
    return set(text1.split()).intersection(set(text2.split()))


#####计算各种距离和相似性
#计算jacard距离
def jac_dis(text1, text2):
    #text1, text2 = text.split('\t')
    ls_1 , ls_2 = set(text1.split()), set(text2.split())
    return len(ls_1.intersection(ls_2)) / len(ls_1.union(ls_2)) * 1.0

#编辑(Levenshtein)距离
from Levenshtein import *

def Leven_ratio(text1, text2):
    #text1, text2 = text.split('\t')
    return ratio(text1, text2)

def Leven_dis(text1, text2):
    #text1, text2 = text.split('\t')
    return distance(text1, text2)

##fuzz距离
def fuzz_ratio(text1, text2):
    #text1, text2 = text.split('\t')
    return fuzz.ratio(text1, text2)


def fpartial_ratio(text1, text2):
    #text1, text2 = text.split('\t')
    return fuzz.partial_ratio(text1, text2)

def ftoken_sort_ratio(text1, text2):
    #text1, text2 = text.split('\t')
    return fuzz.token_sort_ratio(text1, text2)


def ftoken_set_ratio(text1, text2):
    #text1, text2 = text.split('\t')
    return fuzz.token_set_ratio(text1, text2)

#计算tfidf相似度
def cos_similar(text1, text2, tfidf_model):
    #text1, text2 = text.split('\t')
    vec1 = tfidf_model.transform([text1]).toarray()
    vec2 = tfidf_model.transform([text2]).toarray()
    return (vec1 * vec2).sum() / sqrt(power(vec1, 2).sum()) * sqrt(power(vec2, 2).sum())
    # return cosine_similarity(vec1, vec2)


def cos_similar_lda(text1, text2, tfidf_model, lda_model):
    #text1, text2 = text.split('\t')
    vec1 = lda_model.transform(tfidf_model.transform([text1]))
    vec2 = lda_model.transform(tfidf_model.transform([text2]))
    return (vec1 * vec2).sum() / sqrt(power(vec1, 2).sum()) * sqrt(power(vec2, 2).sum())
    #return cosine_similarity(vec1, vec2)

#求每句话的平均词向量
def avg_vec(text):
    values = text.split(' ')
    ls = []
    for value in values:
        if text.find('W')>=0:
            ls.append(embeddings_word[value])
        else:
            ls.append(embeddings_char[value])
    return average(array(ls)).reshape(-1,1)

#计算词向量的相似度
def vec_similar(text):
    text1, text2 = text.split('\t')
    vec1 = abs(avg_vec(text1))
    vec2 = abs(avg_vec(text2))
    return (vec1 * vec2).sum() / sqrt(power(vec1, 2).sum()) * sqrt(power(vec2, 2).sum())
    #return cosine_similarity(vec1, vec2)


#把词或字符变成对应的数字
def text2int(text, vocab):
    if text is not np.nan:
        return [vocab[value] for value in text.split()]
    else:
        return []