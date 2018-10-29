# -*- coding:utf-8 -*-

from pyhanlp import *
import jieba
import jieba.posseg as pseg
from preprocess import *



###不保留词性(分去停用词和不去停用词)
#分词不去停用词
def seg_cut(text, path, pos, stop_words):
    if path == False:
        if pos == True:
            return ' '.join([str(term) for term in HanLP.segment(text.strip())])
        else:
            return ' '.join([str(term).split('/')[0] for term in HanLP.segment(text.strip())])
    else:
        return ' '.join([str(term).split('/')[0] for term in HanLP.segment(text.strip()) \
                         if str(term).split('/')[0] not in stop_words])


import re, sys, unicodedata


res = re.compile(r'\s+')
red = re.compile(r'^(\d+)$')
# 清洗标点符号等异常字符
todel = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)) not in ('Lu', 'Ll', 'Lt', 'Lo', 'Nd', 'Nl', 'Zs'))

# 清洗分词结果的方法
def cleantext(text):
    try:
        text = unicode(text)
    except:
        pass
    if text != '':
        return re.sub(res, ' ', ' '.join(map(lambda x: re.sub(red, '', x), text.translate(todel).split(' ')))).strip()
    else:
        return text


text = "积满38个赞参与健康大礼包活动、即可免费领取价值1180元养生调理蚕丝被一床，你吃饭了吗？"
stop_words = load_data("../dict/stop_word.txt")
print(seg_cut(text, False, True, stop_words))
cut = seg_cut(text, True, False, stop_words)
print(cut)
cut_clean = cleantext(cut)
print(cut_clean)
