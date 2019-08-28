# 毕设
# 设置numpy和Tensorflow的随机种子，置顶
from numpy.random import seed
seed(1337)
from tensorflow import set_random_seed
set_random_seed(1337)

# set GPU memory
import keras.backend as K
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.4   # 按比例
    sess = tf.Session(config=config)

'''
实体上下文表示学习c：
1、CNN+attention
2、单层神经网络 f=（词向量平均+拼接+tanh+Dropout）

候选选择：
1、Local modeling 方法
计算所有候选candidate与上下文的相似度，
并对<m,c1>...<m,cx>的得分进行排序 ranking
得分最高者作为mention的id
2、拼接【候选，上下文表示，相似度得分】，softmax分类

组成：
semantic representation layer
convolution layer
pooling layer
concatenation layer (Vm + Vc + Vsim)    Vsim=Vm·M·Vc
hidden layer
softmax layer (0/1)

参考 BMC Bioinformatics
《CNN-based ranking for biomedical entity normalization》
《ACL2018-Linking 》
'''
import os
import time
import csv
import pickle as pkl
from keras.layers import *
from keras.models import Model, load_model
from keras.utils import plot_model, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.callbacks import Callback
from keras.regularizers import l2
from keras_layer_normalization import LayerNormalization
# import importlib
# m=importlib.import_module("sample._test_nnet")
import sys
sys.path.append("..")
from utils.attention_keras import Attention, Position_Embedding
import _test_nnet as m
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import sys
# print(sys.version)
# if sys.version.startswith('3.6'):
#     import tensorflow_hub as hub
#     # Now instantiate the elmo model
#     elmo_model = hub.Module("temp/moduleA", trainable=True)
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.tables_initializer())

# Parameters of the network
word_emb_size = 200
dropout_rate = 0.5  # [0.5, 0.5]
num_classes = 2

lstm_size = [200]    # BLSTM 隐层大小
feature_maps = [25, 25]
kernels = [2, 3]

epochs = 4
batch_size = 64 # 8 64
learning_rate = 1e-4   # 5e-4 1e-4
decay_rate = 0.0    # 完全禁用梯度衰减 learning_rate / epochs
optimizer = 'adagrad'  # adagrad(效果最好)

max_match = 0
context_window_size = 10
max_seq_length = 427
print('修改句子最大长度:427')

nn = [['swem-aver', 'swem-max', 'bow'],
      ['bilstm', 'bilstm-max', 'bilstm-mean'], 
      ['cnn', 'hierarchical-convnet'], 
      ['transformer', 'transformerQ', 'transformer-lstm', 'transformer-cnn']]
shared_context_network = nn[0][1]
merge_layer = ['vector', 'attention', 'bi-attention'][0]

weights_file = None
rootCorpus = r'data'
embeddingPath = r'embedding'


def my_dot(inputs):
    a = inputs[0] * inputs[1]
    a = K.sum(a, axis=-1, keepdims=True)
    a = K.sigmoid(a)
    # a = K.softmax(a)   # 预测结果全部趋于0，别用softmax?
    return a


def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n
    return x_mean


def to_prob(input):
    sum = K.sum(input, 1, keepdims=True)
    return input / sum


def sum_seq(x):
    return K.sum(x, axis=1, keepdims=False)


def max(x):
    x_max = K.max(x, axis=1, keepdims=False)
    return x_max


def buildAttention3(seq, controller):
    controller_repeated = RepeatVector(context_window_size)(controller)
    attention = merge([controller_repeated, seq], mode='concat', concat_axis=-1)

    attention = Dense(1, activation='softmax')(attention)
    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply([attention_repeated, seq])
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)

    return summed, attention


def buildAttention2(seq, controller):
    controller_repeated = RepeatVector(context_window_size)(controller)
    controller = TimeDistributed(Dense(1))(controller_repeated)
    seq1 = TimeDistributed(Dense(1))(seq)
    attention = Add()([controller, seq1])

    attention = Permute((2, 1))(attention)
    # attention = Dense(1, activation='softmax')(attention)
    attention = Dense(context_window_size, activation='softmax')(attention)
    attention = Flatten()(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    output_attention_mul = multiply(inputs=[seq, attention_repeated])
    summed = Lambda(sum_seq, output_shape=(200,))(output_attention_mul)

    return summed, attention


def buildAttention(seq, controller):

    controller_repeated = RepeatVector(2*context_window_size)(controller)
    controller_repeated = TimeDistributed(Dense(200))(controller_repeated)

    attention = Lambda(my_dot, output_shape=(context_window_size,))([controller_repeated, seq])

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(2*context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply(inputs=[attention_repeated, seq])
    # weighted = merge([attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)
    return summed, attention


def self_attention(seq):

    dim = int(seq.shape[-1])
    attention = Lambda(my_dot, output_shape=(2*context_window_size,))([seq, seq])

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(2*context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply(inputs=[attention_repeated, seq])
    # weighted = merge([attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)
    return summed, attention


def CNN(concat_input):
    shared_layer1 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    shared_layer2 = Conv1D(200, kernel_size=4, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    shared_layer3 = Conv1D(200, kernel_size=5, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    output = shared_layer1(concat_input)
    output = BatchNormalization(momentum=0.8)(output)
    output = shared_layer2(output)
    output = BatchNormalization(momentum=0.8)(output)
    output = shared_layer3(output)

    # output = MaxPooling1D(pool_size=10)(output)   # 加入MaxPooling1D效果降低
    # output = Flatten()(output)

    return output


def hier_CNN(concat_input):
    left_cnn1 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(concat_input)
    left_max1 = GlobalMaxPool1D()(left_cnn1)

    left_cnn2 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn1)
    left_max2 = GlobalMaxPool1D()(left_cnn2)

    left_cnn3 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn2)
    left_max3 = GlobalMaxPool1D()(left_cnn3)

    left_cnn4 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn3)
    left_max4 = GlobalMaxPool1D()(left_cnn4)

    return concatenate([left_max1, left_max2, left_max3, left_max4], axis=1)


def hier_CNN_context(concat_input):
    left_cnn1 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(concat_input)
    if merge_layer == 'vector':
        left_max1 = GlobalMaxPool1D()(left_cnn1)

    left_cnn2 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn1)
    if merge_layer == 'vector':
        left_max2 = GlobalMaxPool1D()(left_cnn2)

    left_cnn3 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn2)
    if merge_layer == 'vector':
        left_max3 = GlobalMaxPool1D()(left_cnn3)

    left_cnn4 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn3)
    if merge_layer == 'vector':
        left_max4 = GlobalMaxPool1D()(left_cnn4)

    if merge_layer == 'vector':
        return concatenate([left_max1, left_max2, left_max3, left_max4], axis=-1)
    else:
        return concatenate([left_cnn1, left_cnn2, left_cnn3, left_cnn4], axis=-1)


def transformerEncoder(input_embed, p_layer, att_layer, dense_layer):
    input_pst = p_layer(input_embed) # 增加Position_Embedding能轻微提高准确率

    # block1
    input_rep = att_layer([input_pst, input_pst, input_pst])
    input_rep = add([input_rep, input_pst])
    input_rep = LayerNormalization()(input_rep)
    input_ff = dense_layer(input_rep)
    input_ff = add([input_ff, input_rep])
    input_ff = LayerNormalization()(input_ff)
    # input_ff = GlobalAveragePooling1D()(input_ff)

    # block2
    input_rep2 = att_layer([input_ff, input_ff, input_ff])
    input_rep2 = add([input_rep2, input_ff])
    input_rep2 = LayerNormalization()(input_rep2)
    input_ff2 = dense_layer(input_rep2)
    input_ff2 = add([input_ff2, input_rep2])
    input_ff2 = LayerNormalization()(input_ff2)
    input_ff2 = GlobalAveragePooling1D()(input_ff2)
    return input_ff2


def transformerEncoder_context(input_embed, p_layer, att_layer, dense_layer):
    input_pst = p_layer(input_embed) # 增加Position_Embedding能轻微提高准确率

    # block1
    input_rep = att_layer([input_pst, input_pst, input_pst])
    input_rep = add([input_rep, input_pst])
    input_rep = LayerNormalization()(input_rep)
    input_ff = dense_layer(input_rep)
    input_ff = add([input_ff, input_rep])
    input_ff = LayerNormalization()(input_ff)

    # block2
    input_rep2 = att_layer([input_ff, input_ff, input_ff])
    input_rep2 = add([input_rep2, input_ff])
    input_rep2 = LayerNormalization()(input_rep2)
    input_ff2 = dense_layer(input_rep2)
    input_ff2 = add([input_ff2, input_rep2])
    input_ff2 = LayerNormalization()(input_ff2)

    if merge_layer=='vector':
        input_ff2 = GlobalAveragePooling1D()(input_ff2)
    return input_ff2


def build_model():
    word_embed_layer = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                                 output_dim=embedding_matrix.shape[1],  # 词向量的维度
                                 input_length=context_window_size,
                                 weights=[embedding_matrix],
                                 trainable=True,
                                 name='word_embed_layer')
    candidate_embed_layer = Embedding(input_dim=conceptEmbeddings.shape[0],
                                      output_dim=conceptEmbeddings.shape[1],
                                      input_length=1,
                                      weights=[conceptEmbeddings],    # AutoExtend 训练获得
                                      trainable=True,
                                    name='candidate_embed_layer')
    # id描述专用
    candidate_def_embed_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                            output_dim=embedding_matrix.shape[1],
                                            input_length=max_len_def-1,
                                            weights=[embedding_matrix],    # AutoExtend 训练获得
                                            trainable=True,
                                            name='id_embed_layer')

    model_input = []

    # 候选ID
    candidate_input1 = Input(shape=(1,), dtype='int32', name='candidate_input')
    candidate_embed1 = candidate_embed_layer(candidate_input1)
    # candidate_embed1 = Flatten()(candidate_embed1)

    model_input.append(candidate_input1)
    candidate_embed = candidate_embed1

    # # 候选id描述文本输入
    # candidate_input2 = Input(shape=(max_len_def-1,), dtype='int32', name='id_input')
    # candidate_embed2 = candidate_def_embed_layer(candidate_input2)
    # p_layer = Position_Embedding()
    # att_layer = Attention(10,20)
    # dense_layer = Dense(200, activation='relu')
    # candidate_embed2 = transformerEncoder(candidate_embed2, p_layer, att_layer, dense_layer)
    # candidate_embed2 = RepeatVector(1)(candidate_embed2)    # (?, 1, 200)
    # model_input.append(candidate_input2)
    # # candidate_embed = add([candidate_embed1, candidate_embed2])
    # candidate_embed = candidate_embed2
    # # cnn_layer = Conv1D(200, kernel_size=3, activation='relu', padding='same',
    # #                        kernel_regularizer=l2(1e-4),
    # #                        bias_regularizer=l2(1e-4))
    # # candidate_embed = cnn_layer(candidate_embed2)
    # # candidate_embed = GlobalMaxPool1D()(candidate_embed)   # 加入MaxPooling1D效果降低

    # addContextInput

    left_context_input = Input(shape=(context_window_size,), dtype='int32', name='left_context_input')
    left_context_embed = word_embed_layer(left_context_input)
    model_input.append(left_context_input)

    right_context_input = Input(shape=(context_window_size,), dtype='int32', name='right_context_input')
    right_context_embed = word_embed_layer(right_context_input)
    model_input.append(right_context_input)
    
    # 拼接左右上下文，形成上下文句子输入
    context_embed = concatenate([left_context_embed, right_context_embed], axis=1)
    context_embed = Dropout(0.5)(context_embed)  
    # context_embed = BatchNormalization()(context_embed)  
    
    # 基于简单向量的句子编码器
    if shared_context_network == 'swem-aver':
        shared_layer = GlobalAveragePooling1D()
        candidate_rep = shared_layer(candidate_embed)
        context_rep = shared_layer(context_embed)
    elif shared_context_network == 'swem-max':
        shared_layer = GlobalMaxPooling1D()
        candidate_rep = shared_layer(candidate_embed)
        context_rep = shared_layer(context_embed)
    elif shared_context_network == 'bow':
        shared_layer = Lambda(sum_seq, output_shape=(200,))
        candidate_rep = shared_layer(candidate_embed)
        context_rep = shared_layer(context_embed)
    
    # 基于LSTM的句子编码器
    elif shared_context_network=='gru':
        shared_layer = CuDNNGRU(200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))
        candidate_rep = shared_layer(candidate_embed)
        context_rep = shared_layer(context_embed)
    elif shared_context_network=='bilstm':
        shared_layer = Bidirectional(CuDNNLSTM(200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)))
        candidate_rep = shared_layer(candidate_embed)
        context_rep = shared_layer(context_embed)
    elif shared_context_network == 'bilstm-max':
        shared_layer = Bidirectional(CuDNNLSTM(units=200,
                                               return_sequences=True,
                                               kernel_regularizer=l2(1e-4),
                                               bias_regularizer=l2(1e-4)))
        candidate_rep = shared_layer(candidate_embed)
        candidate_rep = GlobalMaxPooling1D()(candidate_rep)
        context_rep = shared_layer(context_embed)
        if merge_layer == 'vector':
            context_rep = GlobalMaxPooling1D()(context_rep)
    elif shared_context_network == 'bilstm-mean':
        shared_layer = Bidirectional(CuDNNLSTM(units=200,
                                               return_sequences=True,
                                               kernel_regularizer=l2(1e-4),
                                               bias_regularizer=l2(1e-4)))
        candidate_rep = shared_layer(candidate_embed)
        candidate_rep = GlobalAveragePooling1D()(candidate_rep)
        context_rep = shared_layer(context_embed)
        if merge_layer == 'vector':
            context_rep = GlobalAveragePooling1D()(context_rep)

    # 基于CNN的句子编码器
    elif shared_context_network=='cnn':
        shared_layer = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
        candidate_rep = shared_layer(candidate_embed)
        candidate_rep = GlobalMaxPool1D()(candidate_rep)
        context_rep = shared_layer(context_embed)
        if merge_layer == 'vector':
            context_rep = GlobalMaxPool1D()(context_rep)
    elif shared_context_network=='hierarchical-convnet':
        # shared_layer = hier_CNN
        candidate_rep = hier_CNN(candidate_embed)
        candidate_rep = Dense(200)(candidate_rep)
        context_rep = hier_CNN_context(context_embed)
        context_rep = Dense(200)(context_rep)

    # 基于Transformer的句子编码器
    elif shared_context_network == 'transformer':
        p_layer = Position_Embedding()
        att_layer = Attention(10,20)
        dense_layer = Dense(200, activation='relu')
        candidate_rep = transformerEncoder(candidate_embed, p_layer, att_layer, dense_layer)
        context_rep = transformerEncoder_context(context_embed, p_layer, att_layer, dense_layer)
    elif shared_context_network == 'transformerQ':
        p_layer = Position_Embedding()
        att_layer = Attention(10,20)
        input_ct = p_layer(context_embed) # 增加Position_Embedding能轻微提高准确率
        input_cd = p_layer(candidate_embed) # 增加Position_Embedding能轻微提高准确率
        candidate_rep = att_layer([input_cd, input_cd, input_cd])
        input_rep = att_layer([candidate_rep, input_ct, input_ct])
        input_rep = add([input_rep, input_ct])  # (200, 200) (20, 200)
        input_rep = LayerNormalization()(input_rep)
        input_ff = Dense(200, activation='relu')(input_rep)
        input_ff = add([input_ff, input_rep])
        input_ff = LayerNormalization()(input_ff)
        input_ff = GlobalAveragePooling1D()(input_ff)
        x = input_ff
    elif shared_context_network == 'transformerT':
        # 先输入候选描述，得到ID表示，重复并拼接给上下文，输入
        p_layer = Position_Embedding()
        att_layer = Attention(10,20)
        dense_layer = Dense(200, activation='relu')
        candidate_rep = transformerEncoder(candidate_embed, p_layer, att_layer, dense_layer)
        candidate_rep_copy = RepeatVector(2*context_window_size)(candidate_rep)
        context_embed = concatenate([context_embed, candidate_rep_copy], axis=1)
        x = transformerEncoder(context_embed, p_layer, att_layer, dense_layer)
    elif shared_context_network == 'transformer-lstm':
        pass
    elif shared_context_network == 'transforme-cnn':
        pass

    # 融合层
    if merge_layer == 'vector':
        # 组合上下文和ID所有相关的信息
        a = multiply([context_rep, candidate_rep])
        b = subtract([context_rep, candidate_rep])
        x = concatenate([context_rep, candidate_rep, a, b])
    elif merge_layer == 'self-attention':
        context_rep, attn_values_left = self_attention(context_rep)
        candidate_rep, attn_values_right = self_attention(candidate_rep)
        # attn += [attn_values_left, attn_values_right]  # attention 的输出
        x = concatenate([context_rep, candidate_rep])
    elif merge_layer == 'attention':
        # candidate_poj = Dense(200, activation='tanh')(candidate_rep)
        context_poj, attn_values_left = buildAttention(context_rep, candidate_rep)
        # attn += [attn_values_left, attn_values_right]  # attention 的输出
        x = concatenate([context_poj, candidate_rep])
    elif merge_layer == 'bi-attention':
        C, attn_values_left = self_attention(context_rep)
        candidate_poj = Dense(200, activation='tanh')(candidate_rep)
        context_poj, attn_values_left = buildAttention(context_rep, candidate_poj)
        x = concatenate([context_poj, candidate_poj, subtract([context_poj, candidate_poj])])
        r = Dense(200, activation='relu')(x)
        g = Dense(200, activation='sigmoid')(x)
        f = Lambda(lambda x: 1-x)
        o = add([multiply([g, r]), multiply([f(g), candidate_poj])])
        x = concatenate([o, C])
    else:
        pass

    x = concatenate([x, candidate_embed1])

    # build classifier model
    x = Dense(200, activation='relu', name='dense_layer1')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(2, activation='softmax', name='main_output')(x)

    model = Model(inputs=model_input, outputs=[output])
    # attn_model = Model(input=x_input, output=attn)
    if weights_file is not None:
        model.load_weights(weights_file)

    '''
    loss_function: binary_crossentropy / categorical_crossentropy
    decay=lr/nb_epoch 1e-6
    optimizer: 'adam'/'nadam'/'rmsprop':最后所有样例的输出概率都一样? lr过大
                adagrad: 表现较为正常
                adadelta: 偏向于将样例分类为1
    '''

    if optimizer.lower() == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.5, decay=decay_rate)  # Adam对超参数更宽容
    elif optimizer.lower() == 'nadam':
        opt = Nadam(lr=learning_rate, decay=decay_rate)
    elif optimizer.lower() == 'adagrad':
        opt = Adagrad(lr=learning_rate) # 参数推荐使用默认值 , clipvalue=1.
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate, decay=decay_rate)
    elif optimizer.lower() == 'sgd':
        opt = SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
        # opt = SGD(lr=0.001, momentum=0.9, decay=0., nesterov=True, clipvalue=5)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])
    # model.summary()
    print("model compiled!")
    plot_model(model, to_file='data/model.png', show_shapes=True)
    return model


# 自己测
class ConllevalCallback(Callback):
    def __init__(self, X_test, y_test, max_match):
        super(ConllevalCallback, self).__init__()
        self.X = X_test
        self.y = y_test
        self.max_match = max_match

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.X)  # 模型预测
        y_pred = predictions.argmax(axis=-1)  # Predict classes [0]
        y_test = self.y.argmax(axis=-1)
        print(predictions[:20])

        TP = 0
        FP = 0
        FN = 0
        for i in range(len(y_test)):
            if y_test[i]==1 and y_pred[i]==1:
                TP+=1
            elif y_test[i]==0 and y_pred[i]==1:
                FP+=1
            elif y_test[i]==1 and y_pred[i]==0:
                FN+=1

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = (2*P*R)/(P+R)

        with open('prf.txt','a') as f:
            f.write('{}\n{}\t{}\t{}'.format(str(epoch), TP,FP,FN))
            f.write('\n')
            f.write('{}\t{}\t{}'.format(P,R,F))
            f.write('\n')

        if F>self.max_match:
            print('\nTP:{}\tF:{}'.format(TP,F))
            # self.model.save('data/weights_arnn'+'.hdf5')
            self.model.save('ned_model/weights_rnn_{}.hdf5'.format(F))
            self.max_match = F


# 脚本测
class ConllevalCallback2(Callback):
    '''
    Callback for running the conlleval script on the test dataset after each epoch.
    '''
    def __init__(self, max_match, word_index, id2def):
        super(ConllevalCallback2, self).__init__()
        self.max_match = max_match
        self.word_index = word_index # 用于id描述文本
        self.id2def = id2def # 用于id描述文本

    def on_epoch_end(self, epoch, logs={}):
        for prob in [1, 1.2]:
            m.main(self.model, prob, self.word_index, self.id2def, 'result/predictions_76.31.pkl')
            # m.main(self.model, prob, self.word_index, self.id2def, 'result/predictions_bert_best.pkl')
            
            try:
                os.system('python evaluation/BioID_scorer_1_0_3/scripts/bioid_score.py '
                            '--verbose 1 --force '
                            'evaluation/BioID_scorer_1_0_3/scripts/bioid_scores '
                            'evaluation/test_corpus_20170804/caption_bioc '
                            'system1:evaluation/test_corpus_20170804/prediction')
            except expression as identifier:
                raise()
            
            res = []
            csv_file = 'evaluation/BioID_scorer_1_0_3/scripts/bioid_scores/corpus_scores.csv'
            with open(csv_file) as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                birth_header = next(csv_reader)  # 读取第一行每一列的标题
                i = 0
                for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
                    item = []
                    if i in [2, 8, 14, 20]:
                        res.append(row[14:17])
                    if i == 20:
                        res.append(row[17:])
                        break
                    i += 1
            with open('prf.txt', 'a') as f:
                f.write('{} {}\n'.format(shared_context_network, merge_layer))
                f.write('{}'.format(epoch))
                f.write('\n')

                for i in range(len(res)):
                    item = [one[:7] for one in res[i]]  # 取小数点后四位
                    if i == 0:
                        f.write('any, strict: ')
                    if i == 1:
                        f.write('any,overlap: ')
                    if i == 2:
                        f.write('normalized,strict: ')
                    if i == 3:
                        f.write('normalized,overlap: ')
                    f.write('\t'.join(item))
                    f.write('\n')

            last_line = [one[:7] for one in res[-1]]
            print('\t'.join(last_line))
            F = float(last_line[5])
            print(F)
            if F>self.max_match:
                self.model.save('ned/ned_model/weights_ned_max.hdf5', overwrite=True)
                self.model.save_weights('ned/ned_model/weights_ned_max.weights', overwrite=True)
                self.max_match = F


if __name__ == '__main__':

    shared_context_network = sys.argv[1]
    merge_layer = sys.argv[2]

    max_len_def = 200
    with open(embeddingPath + '/emb.pkl', "rb") as f:
        embedding_matrix = pkl.load(f)

    # # 第四章
    # with open('ned/data/data_train_def.pkl', "rb") as f:
    #     x_left, x_pos_left, x_right, x_pos_right, y, x_elmo_l, x_elmo_r = pkl.load(f) # 250560
    # with open('ned/data/data_train_id_def.pkl', "rb") as f:
    #     x_id, x_id_index, id2def, conceptEmbeddings = pkl.load(f)    # ID描述文本，ID索引
    # with open('word_index.pkl', "rb") as f:
    #     word_index = pkl.load(f)
    # from keras.preprocessing.sequence import pad_sequences
    # x_id = pad_sequences(x_id, maxlen=max_len_def-1, padding='post')
    # x_id_index = np.array(x_id_index)

    # 第五章
    with open('ned/data/data_train2.pkl', "rb") as f:
        x_left, x_pos_left, x_right, x_pos_right, y, x_elmo_l, x_elmo_r = pkl.load(f)
    with open('ned/data/id_embedding.pkl', "rb") as f:
        x_id, x_id_test, conceptEmbeddings = pkl.load(f)
    print(conceptEmbeddings.shape)  # (12657, 200) 从1开始编号

    # # test
    # x_id_test = np.array(x_id_test + x_id)
    # x_left_test = np.array(x_left_test + x_left)
    # x_right_test = np.array(x_right_test + x_right)
    # y_test = to_categorical(y_test + y, 2)
    # y_test = np.array(y_test)
    # testSet = [x_id_test, x_left_test, x_right_test]

    ind = len(x_id) // batch_size

    # train
    x_id = np.array(x_id)
    x_left = np.array(x_left)
    x_pos_left = np.array(x_pos_left)
    x_right = np.array(x_right)
    x_pos_right = np.array(x_pos_right)
    y = to_categorical(y[:batch_size * ind], 2)
    y = np.array(y)
    x_elmo_l = np.array(x_elmo_l)
    x_elmo_r = np.array(x_elmo_r)

    # dataSet = [x_id_index[:batch_size * ind], x_id[:batch_size * ind], x_left[:batch_size * ind], x_right[:batch_size * ind]]
    dataSet = [x_id[:batch_size * ind], x_left[:batch_size * ind], x_right[:batch_size * ind]]
    print(x_id.shape, x_left.shape, y.shape)  # (246149, 1) (246149, 10) (246149, 2)

    savedPath = 'data/weights2.{epoch:02d}-{val_ac c:.2f}.hdf5'
    saveModel = ModelCheckpoint(savedPath,
                                monitor='val_acc',
                                save_best_only=True,  # 只保存在验证集上性能最好的模型
                                save_weights_only=False,
                                mode='auto')
    earlyStop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    conllback = ConllevalCallback2(max_match, None, None)
    # conllback = ConllevalCallback2(max_match, word_index, id2def)
    tensorBoard = TensorBoard(log_dir='./ned_model',  # 保存日志文件的地址,该文件将被TensorBoard解析以用于可视化
                              histogram_freq=0)  # 计算各个层激活值直方图的频率(每多少个epoch计算一次)

    start_time = time.time()
    model = build_model()
    model.fit(x=dataSet, y=y,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[conllback],)
              # validation_split=0.1)
    time_diff = time.time() - start_time
    print("Total %.2f min for training" % (time_diff / 60))
    print(max_match)

    # # 加载预训练的模型,再训练
    # root = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/'
    # model = load_model(root + 'ned/ned_model/weights_rnn_0.9412647734274352.hdf5')
    # m.main(model)

'''
python /home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_score.py --verbose 1 --force /home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_scores /home/administrator/桌面/BC6_Track1/test_corpus_20170804/caption_bioc system1:/home/administrator/桌面/BC6_Track1/test_corpus_20170804/prediction
'''

