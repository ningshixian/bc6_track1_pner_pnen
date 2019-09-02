'''
BMC BioInformatics
BLSTM-CRF --> BioID track1

思路：
1、转换为3tag标注问题（0：非实体，1：实体的首词，2：实体的内部词）；
2、获取对应输入的语言学特征（字符特征，词性，chunk，词典特征，大小写）
3、通过双向LSTM，直接对输入序列进行概率预测
4、通过CRF+viterbi算法获得最优标注结果；
'''

# 设置numpy和Tensorflow的随机种子，置顶
from numpy.random import seed
seed(1337)
from tensorflow import set_random_seed
set_random_seed(1337)

import os
import random
import math
import pickle as pkl
import string
import time
from tqdm import tqdm
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import *
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.optimizers import *
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
# from keras_contrib.layers import CRF
from keraslayers.ChainCRF import ChainCRF
# from sample.keraslayers.crf_keras import CRF
from utils.helpers import createCharDict
from utils.callbacks import ConllevalCallback
import numpy as np
import codecs
from math import ceil
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from collections import OrderedDict
import tensorflow_hub as hub
import keras.backend as K

# set GPU memory
if 'tensorflow'==K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    # # 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # 按需求增长
    # sess = tf.Session(config=config)
    # # set_session(sess)

    # 方法2:只允许使用x%的显存,其余的放着不动
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5    # 按比例
    sess = tf.Session(config=config)


# Parameters of the network
word_emb_size = 200
char_emb_size = 50
cap_emb_size = 10
pos_emb_size = 25
chunk_emb_size = 10
dict_emb_size = 15
num_classes = 5

epochs = 15
batch_size = 8 # 32
dropout_rate = 0.5  # [0.5, 0.5]
optimizer = 'rmsprop'  # 'rmsprop'
learning_rate = 1e-3    # 1e-3  5e-4
decay_rate = learning_rate / epochs     # 1e-6

# BLSTM 隐层大小
lstm_size = [200]    
# CNN settings
feature_maps = [25, 25]
kernels = [2, 3]

max_f = 0
use_chars = True
batch_normalization = False
highway = False



def _shared_layer(concat_input):
    '''共享不同任务的Embedding层和bilstm层'''
    cnt = 0
    for size in lstm_size:
        cnt += 1
        if isinstance(dropout_rate, (list, tuple)):
            output = Bidirectional(LSTM(units=size,
                                        return_sequences=True,
                                        dropout=dropout_rate[0],
                                        recurrent_dropout=dropout_rate[1],
                                        # stateful=True, # 上一个batch的最终状态作为下个batch的初始状态
                                        kernel_regularizer=l2(1e-4),
                                        bias_regularizer=l2(1e-4),
                                        implementation=2),
                                   name='shared_varLSTM_' + str(cnt))(concat_input)
        else:
            """ Naive dropout """
            output = Bidirectional(CuDNNLSTM(units=size,
                                             return_sequences=True,
                                             kernel_regularizer=l2(1e-4),
                                             bias_regularizer=l2(1e-4)),
                                   name='shared_LSTM1_' + str(cnt))(concat_input)
    return output


def CNN(seq_length, length, feature_maps, kernels, x):
    '''字符表示学习'''
    concat_input = []
    for filters, size in zip(feature_maps, kernels):
        charsConv1 = TimeDistributed(Conv1D(filters, size, padding='same', activation='relu'))(x)
        charsPool = TimeDistributed(GlobalMaxPool1D())(charsConv1)
        # reduced_l = length - kernel + 1
        # conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
        # maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
        concat_input.append(charsPool)

    x = Concatenate()(concat_input)
    x = Reshape((seq_length, sum(feature_maps)))(x)
    return x


# Now instantiate the elmo model
elmo_model = hub.Module("embedding/moduleA", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[sentence_maxlen])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


TIME_STEPS = 21
def buildModel():
    char2idx = createCharDict()

    char_embedding = np.zeros([len(char2idx)+1, char_emb_size])
    for key, value in char2idx.items():
        limit = math.sqrt(3.0 / char_emb_size)
        vector = np.random.uniform(-limit, limit, char_emb_size)
        char_embedding[value] = vector

    '''字向量,若为shape=(None,)则代表输入序列是变长序列'''
    tokens_input = Input(shape=(sentence_maxlen,), name='tokens_input', dtype='int32')  # batch_shape=(batch_size,
    tokens_emb = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                           output_dim=embedding_matrix.shape[1],  # 词向量的维度
                           weights=[embedding_matrix],
                           trainable=True,
                           # mask_zero=True,    # 若√则编译报错，CuDNNLSTM 不支持？
                           name='token_emd')(tokens_input)

    '''字符向量'''
    chars_input = Input(shape=(sentence_maxlen, word_maxlen,), name='chars_input', dtype='int32')
    chars_emb = TimeDistributed(Embedding(input_dim=char_embedding.shape[0],
                                          output_dim=char_embedding.shape[1],
                                          weights=[char_embedding],
                                          trainable=True,
                                          # mask_zero=True,
                                          name='char_emd'))(chars_input)
    chars_emb = CNN(sentence_maxlen, word_maxlen, feature_maps, kernels, chars_emb)

    mergeLayers = [tokens_emb, chars_emb]

    # Additional features
    cap_input = Input(shape=(sentence_maxlen,), name='cap_input')
    cap_emb = Embedding(input_dim=7,  # 索引字典大小
                        output_dim=cap_emb_size,  # 大小写特征向量的维度
                        trainable=True)(cap_input)
    mergeLayers.append(cap_emb)

    pos_input = Input(shape=(sentence_maxlen,), name='pos_input')
    pos_emb = Embedding(input_dim=60, 
                        output_dim=pos_emb_size,  # pos特征向量的维度
                        trainable=True)(pos_input)
    mergeLayers.append(pos_emb)

    chunk_input = Input(shape=(sentence_maxlen,), name='chunk_input')
    chunk_emb = Embedding(input_dim=25,
                          output_dim=chunk_emb_size,  # chunk特征向量的维度
                          trainable=True)(chunk_input)
    mergeLayers.append(chunk_emb)

    # 加入词典特征
    dict_input = Input(shape=(sentence_maxlen,), name='dict_input')
    dict_emb = Embedding(input_dim=5,
                          output_dim=dict_emb_size,  # dict特征向量的维度
                          trainable=True)(dict_input)
    mergeLayers.append(dict_emb)

    # 加入ELMO向量
    elmo_input = Input(shape=(sentence_maxlen,), dtype=K.dtype(K.placeholder(dtype=tf.string)))
    # elmo_input = Input(shape=(sentence_maxlen,), dtype=tf.string)
    elmo_emb= Lambda(ElmoEmbedding, output_shape=(sentence_maxlen, 1024))(elmo_input)
    mergeLayers.append(elmo_emb)

    concat_input = concatenate(mergeLayers, axis=-1)  # 拼接 (none, none, 200)
    # concat_input = concatenate([concat_input, elmo_embedding], axis=-1)  # (none, none, 200)

    if batch_normalization:
        concat_input = BatchNormalization()(concat_input)
    if highway:
        for l in range(highway):
            concat_input = TimeDistributed(Highway(activation='tanh'))(concat_input)

    # Dropout on final input
    concat_input = Dropout(dropout_rate)(concat_input)

    # shared layer
    output = _shared_layer(concat_input)  # (none, none, 200)

    # ======================================================================= #

    output = TimeDistributed(Dense(lstm_size[-1], activation='relu', name='relu_layer'))(output)
    output = Dropout(dropout_rate)(output)
    output = TimeDistributed(Dense(num_classes, name='final_layer'))(output)     # 不加激活函数，否则预测结果有问题222222

    # crf = CRF(num_classes, sparse_target=False)  # 定义crf层，参数为True，自动mask掉最有一个标签
    # output = crf(output)  # 包装一下原来的tag_score
    # loss_function = crf.loss_function

    crf = ChainCRF(name='CRF')
    output = crf(output)
    loss_function = crf.loss

    if optimizer.lower() == 'adam':
        opt = Adam(lr=learning_rate, clipvalue=1.)
    elif optimizer.lower() == 'nadam':
        opt = Nadam(lr=learning_rate, clipvalue=1.)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate, clipvalue=1.)
    elif optimizer.lower() == 'sgd':
        opt = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
        # opt = SGD(lr=0.001, momentum=0.9, decay=0., nesterov=True, clipvalue=5)

    model_input =[tokens_input, chars_input, cap_input, pos_input, chunk_input, dict_input, elmo_input]
    model = Model(inputs=model_input, outputs=[output])
    model.compile(loss=loss_function,    # 'categorical_crossentropy'
                  optimizer=opt,
                  metrics=["accuracy"])     # crf.accuracy
    model.summary()

    # plot_model(model, to_file='result/model.png', show_shapes=True)
    return model



if __name__ == '__main__':

    # # 数据预处理
    # import preprocess
    # preprocess.main()

    rootCorpus = r'data'
    embeddingPath = r'embedding'
    idx2label = {0: 'O', 1: 'B-protein', 2: 'I-protein', 3: 'B-gene', 4: 'I-gene'}

    print('Loading data...')

    with open(rootCorpus + '/train.pkl', "rb") as f:
        train_x, train_elmo, train_y, train_char, train_cap, train_pos, train_chunk, train_dict = pkl.load(f)
    with open(rootCorpus + '/test.pkl', "rb") as f:
        test_x, test_elmo, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)
    with open(embeddingPath + '/emb.pkl', "rb") as f:
        embedding_matrix = pkl.load(f)
    with open(embeddingPath + '/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    sentence_maxlen = 427
    print('修改句子最大长度:427')

    dataSet = OrderedDict()
    dataSet['train'] = [train_x, train_cap, train_pos, train_chunk, train_dict]
    dataSet['test'] = [test_x, test_cap, test_pos, test_chunk, test_dict]

    # pad the sequences with zero
    for key, value in dataSet.items():
        for i in range(len(value)):
            dataSet[key][i] = pad_sequences(value[i], maxlen=sentence_maxlen, padding='post')

    # pad the elmo sequences with __PAD__ list
    new_train_elmo = []
    for seq in train_elmo:
        new_seq = []
        for i in range(sentence_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_train_elmo.append(new_seq)

    new_test_elmo = []
    for seq in test_elmo:
        new_seq = []
        for i in range(sentence_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_test_elmo.append(new_seq)

    # pad the char sequences with zero list
    train_char = train_char
    for j in range(len(train_char)):
        train_char[j] = train_char[j][:sentence_maxlen]
        if len(train_char[j]) < sentence_maxlen:
            train_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(train_char[j]))))
    for j in range(len(test_char)):
        test_char[j] = test_char[j][:sentence_maxlen]
        if len(test_char[j]) < sentence_maxlen:
            test_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(test_char[j]))))

    dataSet['train'].insert(1, train_char)
    dataSet['test'].insert(1, test_char)
    dataSet['train'].append(new_train_elmo)
    dataSet['test'].append(new_test_elmo)

    '''
    InvalidArgumentError (see above for traceback): Incompatible shapes: [32,350,5] vs. [8,350,5]
    这是加入ELMO出现的问题,ElmoEmbedding(x)要求输入shape必须满足一个batch的大小
    如果数据集包括32个句子, 其中7个(20%)作为验证集, 另外25个句子训练, 则不满足ELMO的条件!
    '''

    # 截断数据的数量直至可被batch_size整除
    # end1=2
    end1 = len(dataSet['train'][0]) // batch_size
    end2 = len(dataSet['test'][0]) // batch_size
    for i in range(len(dataSet['train'])):
        dataSet['train'][i] = np.array(dataSet['train'][i][:end1 * batch_size])

    for i in range(len(dataSet['test'])):
        dataSet['test'][i] = np.array(dataSet['test'][i])

    train_y = pad_sequences(train_y[:end1 * batch_size], maxlen=sentence_maxlen, padding='post')
    test_y = pad_sequences(test_y, maxlen=sentence_maxlen, padding='post')

    print(dataSet['train'][0].shape)  # (13696, 400)
    print(dataSet['train'][1].shape)  # (13696, 400, 21)
    print(train_y.shape)  # (13696, 400, 5)

    print(dataSet['test'][0].shape)  # (4544, 400)
    print(dataSet['test'][1].shape)  # (4544, 400, 21)
    print(test_y.shape)  # (4544, 400, 5)


    print('done! Model building....')
    model = buildModel()

    calculatePRF1 = ConllevalCallback(dataSet['test'], test_y, 0, idx2label, sentence_maxlen, max_f, batch_size)
    filepath = 'model/weights1.{epoch:02d}-{val_loss:.2f}.hdf5'
    saveModel = ModelCheckpoint(filepath,
                                monitor='val_loss',
                                save_best_only=True,    # 只保存在验证集上性能最好的模型
                                save_weights_only=False,
                                mode='auto')
    earlyStop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    tensorBoard = TensorBoard(log_dir='./model',     # 保存日志文件的地址,该文件将被TensorBoard解析以用于可视化
                              histogram_freq=0)     # 计算各个层激活值直方图的频率(每多少个epoch计算一次)

    # 模型训练
    start_time = time.time()
    model.fit(x=dataSet['train'], y=train_y,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[calculatePRF1])
            #   validation_data=(dataSet['test'], test_y))
    time_diff = time.time() - start_time
    print("%.2f sec for training (4.5)" % time_diff)
    print(model.metrics_names)


    # # 加载模型进行预测,保存识别结果
    # path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/model/Model_Best.h5'
    # if os.path.exists(path):
    #     model.load_weights(path)
    #     print('模型加载成功')
    #     predictions = model.predict(dataSet['test'], batch_size=batch_size, verbose=1)
    #     y_pred = predictions.argmax(axis=-1)
    #     print(len(test_x))
    #     y_pred = y_pred[:len(test_x)]

    #     with open('result/predictions.pkl', "wb") as f:
    #         pkl.dump((y_pred), f, -1)

    #     # from sample.utils.write_test_result import writeOutputToFile
    #     # writeOutputToFile(r'data/test.final.txt', y_pred)


    '''
    通过下面的命令启动 TensorBoard
    tensorboard --logdir=/home/administrator/PycharmProjects/keras_bc6_track1/sample/model
    http://localhost:6006
    '''
