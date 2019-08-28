#! -*- coding: utf-8 -*-
import platform

print('当前python版本号：' + platform.python_version())

from keras import backend as K
from keras.engine.topology import Layer


class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3])
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)




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

    controller_repeated = RepeatVector(context_window_size)(controller)
    controller_repeated = TimeDistributed(Dense(200))(controller_repeated)

    attention = Lambda(my_dot, output_shape=(context_window_size,))([controller_repeated, seq])

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply(inputs=[attention_repeated, seq])
    # weighted = merge([attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)
    return summed, attention


def self_attention(seq):
    # Self-Attention
    dim = int(seq.shape[-1])
    attention = Lambda(my_dot, output_shape=(context_window_size,))([seq, seq])

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(dim)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply(inputs=[attention_repeated, seq])
    # weighted = merge([attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(dim,))(weighted)
    return summed, attention

