import jieba
import codecs
import numpy as np
import re
from zhon.hanzi import punctuation
import string
import time
import os
import collections
from langconv import *
from operator import itemgetter

from model_seq2seq_contrib import Seq2seq

import tensorflow as tf

import config_web


tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

words = []

counter = collections.Counter()

def get_vocab(path):
    raw_data = []
    with codecs.open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            sentence = ''
            for lin in line.split():
                if len(lin) > 1 and lin != '\n':
                    sent = remove_punc(lin)
                    sentence = sentence + sent + ' '
                    for word in cut_sentence(sent):
                        counter[word] += 1
            raw_data.append(sentence)

    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]
    sorted_words = ['<unk>', '<sos>', '<pad>', '<eos>'] + sorted_words

    if len(sorted_words) > config_web.VOCAB_SIZE:
        sorted_words = sorted_words[:config_web.VOCAB_SIZE]

    with codecs.open(config_web.VOCAB_PATH, 'w', encoding='utf-8') as vocab:
        for wor in sorted_words:
            vocab.write(wor + '\n')

    with codecs.open(config_web.RAW_DATA_SIMPLE, 'w', encoding='utf-8') as data:
        for sen in raw_data:
            data.write(sen + '\n')

def cut_sentence(sen):
    return [i for i in jieba.cut(sen) if i != '\n' and i != ' ']


def remove_punc(sen):
    sentence = re.sub(r'[%s]' % punctuation, '', sen)
    punct = set(string.punctuation)
    sentence = ''.join(pu for pu in sentence if pu not in punct)
    sentence = chs_to_simple(sentence)
    return sentence

# 繁体字到简体字的转换
def chs_to_simple(sentence):
    sen = Converter('zh-hans').convert(sentence)
    sen.encode('utf-8')
    return sen


if __name__=='__main__':
    #print([i  for i in jieba.cut('明天你确定犹豫吗')])
    get_vocab(config_web.RAW_DATA_PATH)