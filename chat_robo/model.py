import tensorflow as tf
import config
import data_pro
import sys
import os
import keras
from keras import Sequential
from keras.layers import Embedding, Dropout, Bidirectional, LSTM, Flatten, Dense
from keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq


SOS = '<sos>'
EOS = '<eos>'

class BiModel(object):

    def __init__(self, is_training,batch_size,num_step):
        self.batch_size = batch_size
        self.num_step = num_step
        self.hidden_size = config.HIDDEN_SIZE
        self.embedding_dim = config.EMBEDDING_DIM
        self.num_layers = config.NNUM_LAYERS
        self.max_gard_norm = config.MAX_GRAD_NORM
        self.learning_rate = config.LEARNING_RATE
        self.drop_dense = config.DROP_DENSE
        self.vocab_size = config.VOCAB_SIZE
        self.epoch = config.EPOCH
        self.shared_emb_and_softmax = config.SHARED_EMB_AND_SOFTMAX

        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_step])
        self.target = tf.placeholder(tf.int32, [self.batch_size, self.num_step])
        # self.weight = tf.placeholder(tf.float32,[None,None])

        self.drop_lstm = config.DROP_LSTM if is_training else 1.0
        lstm_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size),output_keep_prob=self.drop_lstm) for _ in range(self.num_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        src_embedding = tf.get_variable('src_embedding', [self.vocab_size, self.hidden_size])
        # self.trg_embedding = tf.get_variable('trg_embedding', [self.vocab_size, self.hidden_size])

        inputs = tf.nn.embedding_lookup(src_embedding, self.input_data)
        # decoder_inputs = tf.nn.embedding_lookup(self.trg_embedding, self.target)

        output = []
        state = self.initial_state

        with tf.variable_scope('RNN'):
            for time in range(self.num_step):
                if time > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time, :], state)

                output.append(cell_output)
        outputs = tf.reshape(tf.concat(output, 1), [-1, self.hidden_size])

        if self.shared_emb_and_softmax:
            self.weight = tf.transpose(src_embedding)
        else:
            self.weight = tf.get_variable('weight', [self.hidden_size, self.vocab_size])
        self.bias = tf.get_variable('bias', [self.vocab_size])

        # input data shape is [batch_size,max_time,depth],max_time is usually a max length of sentence,depth is the dim of word embedding
        self.logits = tf.matmul(outputs, self.weight) + self.bias
        #reshape label[batch, num_steps] to one axis [batch*num_step]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.target, [-1]), logits=self.logits)

        self.cost = tf.reduce_sum(loss) / self.batch_size
        self.final_state = state
        if not is_training:return
        trainable_variables = tf.trainable_variables()

        grad = tf.gradients(self.cost / tf.to_float(self.batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grad, config.MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


def run_epoch(sess, model, batches, train_op, output_log, step,save):
    total_cost = 0.0
    iter = 0
    state = sess.run(model.initial_state)

    for x, y in batches:
        cost, state, _ = sess.run([model.cost, model.final_state, train_op], {
            model.input_data: x, model.target: y, model.initial_state: state
        })

        total_cost += cost
        iter += model.num_step
        if output_log and step % 100 == 0 and step <3001:
            print('after %d step, perplexity is %.3f' % (step, np.exp(total_cost / iter)))

        if step % 300 ==0:
            save.save(sess, config.CHECKPOINT_PATH,global_step=step)
        step += 1

    return step, np.exp(total_cost / iter)

def main():

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    latest = tf.train.latest_checkpoint('./model')
    sd = tf.train.import_meta_graph(latest+'.meta')
    if not latest:
        print('cannot find the model,start training...')

        with tf.variable_scope('language_model', reuse=None, initializer=initializer):
            train_model = BiModel(True,config.BATCH_SIZE,config.MAX_SENTENCE_LENGTH)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            train_Data = data_pro.make_batches()

            step = 0
            for i in range(config.EPOCH):
                saver = tf.train.Saver()
                print('in iteration :  %d ' % (i + 1))
                state, train_pplx = run_epoch(sess, train_model, train_Data, train_model.train_op, True, step,saver)
                print('epoch : %d  train perplexity : %.3f' % (i + 1, train_pplx))

    else:
        print('testing start...')
        sys.stdout.write('ai:')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        if sentence != None:
            sentence = data_pro.remove_punc(sentence).strip()
            sentence = ['<sos>'] + data_pro.cut_sentence(sentence)
            word_2_id = data_pro.get_word_to_id(config.VOCAB_PATH)
            sentence = data_pro.get_pad(sentence) + sentence
            sentence = [data_pro.get_id(i, word_2_id) for i in sentence]
        inputs = tf.reshape(sentence, [1,20])

        pre_model = BiModel(False,1,20)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            sd.restore(sess=sess, save_path=latest)
            tf.global_variables_initializer().run()
            out = sess.run(pre_model.logits,feed_dict={pre_model.input_data:inputs})
            print(out)
            print('load successfully')

def test():
    sys.stdout.write('ai:')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    if sentence != None:
        sentence = data_pro.remove_punc(sentence).strip()
        sentence = ['<sos>'] + data_pro.cut_sentence(sentence)
        word_2_id = data_pro.get_word_to_id(config.VOCAB_PATH)
        sentence = data_pro.get_pad(sentence) + sentence
        sentence = [data_pro.get_id(i,word_2_id) for i in sentence]
    inputs = tf.Variable(sentence)

    latest = tf.train.latest_checkpoint('./model')
    if os.path.exists(latest):
        print('cannot find the model')

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('language_model', reuse=True, initializer=initializer):
        pre_model = BiModel(False,1,20)

    #saver = tf.train.Saver()
    with tf.Session() as sess:
        latest.restore(sess, latest)
        tf.global_variables_initializer().run()
        out = sess.run(pre_model.logits,feed_dict={pre_model.input_data:sentence})
        print(out)
        print('load successfully')





if __name__ == '__main__':
    #test()
    main()

'''
enco,deco = data_pro.make_batches()
model = Sequential()
model.add(Embedding(input_dim=config.VOCAB_SIZE, output_dim=config.EMBEDDING_DIM, input_shape=(20,)))
model.add(Dropout(config.DROP_LSTM))
model.add(Bidirectional(LSTM(config.HIDDEN_SIZE, return_sequences=True), merge_mode='concat'))
model.add(Dropout(config.DROP_LSTM))
model.add(Flatten())
model.add(Dense(20, activation='sigmoid'))

es = EarlyStopping(monitor='val_acc',patience=5)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

for i in range(len(enco)):
    model.fit(enco[i], deco[i], batch_size=100,epochs=5)
'''
