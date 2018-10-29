# -*- coding:utf-8 -*-

import re

def load_data(filepath):
    return [value.replace('\n', '') for value in open(filepath, encoding='utf8').readlines()]


#去除标点符号(多用于语音转成的文本)
def sub_punction(corpus):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", corpus)

#字符
def get_char(text):
    if text != '':
        return ' '.join([value for value in text.replace(' ', '')])


##简繁体转换
from langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


#处理重复字符
class RepeatReplacer(object):
    def __init__(self):
        self.repl = r'\1\2\3'

    def replace_zh(self, word):
        repeat_regexp = re.compile(r'([\u4e00 -\u9fa5]*)([\u4e00 -\u9fa5]+)\2([\u4e00 -\u9fa5]*)')
        repl_word = repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace_zh(repl_word)
        else:
            return repl_word
    
    def replace_en(self, word):
        # if wordnet.synsets(word):
        #     return word
        repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)') #英文去重
        repl_word = repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace_en(repl_word)
        else:
            return repl_word
    
    def replace_string(self, word):
        repeat_regexp = re.compile(r'(.*)(.+)\2(.*)')
        repl_word = repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace_string(repl_word)
        else:
            return repl_word

#中文分句
# cutlist = """
# [。，,！……!《》<>\"':：？\?、\|“”‘’；]{}（）{}【】()｛｝（）：？！。，;、~——+％%`:“”＂'‘\n\r
# """
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')

class SentenceCut(object):
    def sent_cut(self, cutlist, lines):  # 参数1：引用分句标志符；参数2：被分句的文本，为一行中文字符
        l = []  # 句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
        line = []  # 临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空

        for i in range(len(lines)): 
            if lines[i] in cutlist:  # 如果当前字符既不是分句符号，也不是最后一个字符
                line.append(lines[i])  # 将此字符放入临时列表中
                l.append(''.join(line))  # 并把当前临时列表的内容加入到句子列表中
                line = []  # 将符号列表清空，以便下次分句使用
            elif i == len(lines)-1 and lines[i] not in cutlist:
                line.append(lines[i])  # 将此字符放入临时列表中
                l.append(''.join(line))  # 并把当前临时列表的内容加入到句子列表中
                line = []  # 将符号列表清空，以便下次分句使用
            else:
                if i == len(lines)-1:
                    line.append(lines[i])  # 将此字符放入临时列表中
                    l.append(''.join(line))  # 并把当前临时列表的内容加入到句子列表中
                    line = []  # 将符号列表清空，以便下次分句使用
                else:
                    line.append(lines[i])  # 将此字符放入临时列表中
        return l

    def cut_sents(self, cutlist, lines):
        sentences = []
        l = self.sent_cut(list(cutlist), lines)
        for line in l:
            if line.strip() != "":
                li = line.strip().split()
                for sentence in li:
                    if zh_pattern.search(sentence):
                        sentences.append(sentence)
                    else:
                        pass
        return sentences
