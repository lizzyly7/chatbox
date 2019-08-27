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

import config


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

    if len(sorted_words) > config.VOCAB_SIZE:
        sorted_words = sorted_words[:config.VOCAB_SIZE]

    with codecs.open(config.VOCAB_PATH, 'w', encoding='utf-8') as vocab:
        for wor in sorted_words:
            vocab.write(wor + '\n')

    with codecs.open(config.RAW_DATA_SIMPLE, 'w', encoding='utf-8') as data:
        for sen in raw_data:
            data.write(sen + '\n')



def get_pad(sentence):
    if len(sentence) < config.MAX_SENTENCE_LENGTH:
        return ['<pad>'] * (config.MAX_SENTENCE_LENGTH - len(sentence)-1)
    else:
        return []


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


# 对应词汇表，建立词汇表到ID的映射
def get_word_to_id(path):
    with codecs.open(path, 'r', encoding='utf-8') as vocabs:
        vocab = [w.strip() for w in vocabs.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
    return word_to_id


def get_id(word, word_to_id):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


def get_simple_data(path):
    data_train_encoder = []
    encoder_length = []
    decoder_length = []
    data_train_decoder = []
    raw_data_encoder = []
    raw_data_decoder = []
    with codecs.open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if len(line.split()) == 2:
                raw_data_encoder.append(cut_sentence(line.strip().split()[0]))
                raw_data_decoder.append(cut_sentence(line.strip().split()[1]) )
            elif len(line.split()) > 2 and len(line.split()) % 2 == 0:
                num = len(line.split()) % 2
                for i in range(num):
                    raw_data_encoder.append(cut_sentence(line.split()[i]))
                    raw_data_decoder.append(cut_sentence(line.split()[i + 1]) )
            elif len(line.split()) > 2 and len(line.split()) % 2 != 0:
                num1 = len(line.split()) // 2
                for i in range(num1):
                    raw_data_encoder.append(cut_sentence(line.split()[i]))
                    raw_data_decoder.append(cut_sentence(line.split()[i + 1]) )
                raw_data_encoder.append(cut_sentence(line.split()[-1]))
                raw_data_decoder.append(cut_sentence(line.split()[-2]) )

    word_to_id = get_word_to_id(config.VOCAB_PATH)
    for index in range(len(raw_data_decoder)):
        data_all_encoder = raw_data_encoder[index]+ get_pad(raw_data_encoder[index])+['<pad>']
        data_all_decoder = raw_data_decoder[index] + get_pad(raw_data_decoder[index])+['<eos>']

        data_train_encoder.append([get_id(i, word_to_id) for i in data_all_encoder])
        encoder_length.append(len(raw_data_encoder[index]))
        data_train_decoder.append([get_id(i, word_to_id) for i in data_all_decoder])
        decoder_length.append(len(raw_data_decoder[index]))

    return data_train_encoder,encoder_length, data_train_decoder, decoder_length

def make_batches():
    data_en = []
    data_de = []
    enc_length_b = []
    dec_length_b = []
    data_encoder, enc_length, data_decoder, dec_length = get_simple_data(config.RAW_DATA_SIMPLE)

    num_batches = ((len(data_encoder)-1) // config.BATCH_SIZE)
    for i in range(num_batches-1):
        data_en.append(data_encoder[i:config.BATCH_SIZE+i])
        data_de.append(data_decoder[i:config.BATCH_SIZE+i])
        enc_length_b.append(enc_length[i:config.BATCH_SIZE+i])
        dec_length_b.append(dec_length[i:config.BATCH_SIZE+i])

    return data_en, enc_length_b, data_de, dec_length_b


def inference():

    word_2_id = get_word_to_id(config.VOCAB_PATH)
    id_2_word = {k: v for (v, k) in word_2_id.items()}
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    model = Seq2seq(w2i_target=word_2_id, initializer=initializer)
    source_batch, source_lens, target_batch, target_lens = make_batches()

    print_every = 100
    batches = 10400

    with tf.Session() as sess:
        tf.summary.FileWriter('graph', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        losses = []
        total_loss = 0
        try:
            for batch in range(batches):

                feed_dict = {
                    model.seq_inputs: source_batch[batch],
                    model.seq_inputs_length: source_lens[batch],
                    model.seq_targets: target_batch[batch],
                    model.seq_targets_length: target_lens[batch]
                    }

                loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
                total_loss += loss

                if batch % print_every == 0 and batch > 0:
                    print_loss = total_loss if batch == 0 else total_loss / print_every
                    losses.append(print_loss)
                    total_loss = 0
                    print("-----------------------------")
                    print("batch:", batch, "/", batches)
                    print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    print("loss:", print_loss)


            print(losses)
            print(saver.save(sess, "checkpoint/model.ckpt"))
        except Exception as e:

            print(source_batch[batch],np.shape(source_batch[batch]))
            print(source_lens[batch], np.shape(source_lens[batch]))

def test():
    from model_up import Seq2se
    word_2_id = get_word_to_id(config.VOCAB_PATH)
    id_2_word = {k: v for (v, k) in word_2_id.items()}
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    model = Seq2se(w2i_target=word_2_id, initializer=initializer)
    sys.stdout.write('ai:')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    sentence = cut_sentence(remove_punc(sentence))
    print(sentence)
    senten_len = [len(sentence)]
    sentence = sentence+get_pad(sentence)+['<pad>']
    source_sen = [[get_id(i, word_2_id) for i in sentence]]


    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,  "checkpoint/model.ckpt")

        feed_dict = {
            model.seq_inputs: source_sen,
            model.seq_inputs_length: senten_len,
            model.seq_targets: [[0] * config.MAX_SENTENCE_LENGTH],
            model.seq_targets_length: [config.MAX_SENTENCE_LENGTH]
        }
        print(source_sen,np.shape(senten_len))
        print("samples:\n")
        predict_batch = sess.run(model.out, feed_dict)
        for i in range(3):
            print("in:", [id_2_word[num] for num in source_sen[0] if id_2_word[num] != "<pad>" ])
            print("out:", [id_2_word[num] for num in predict_batch[0] if id_2_word[num] != "<pad>"])
            #print("tar:", [id_2_word[num] for num in target_batch[i] if id_2_word[num] != "<pad>"])



if __name__ == '__main__':
    inference()

