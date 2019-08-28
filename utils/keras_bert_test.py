# #! -*- coding:utf-8 -*-

# import json
# import numpy as np
# from random import choice
# from tqdm import tqdm
# from keras_bert import load_trained_model_from_checkpoint, Tokenizer
# import re, os
# import codecs



# mode = 0
# maxlen = 160
# learning_rate = 5e-5
# min_learning_rate = 1e-5

# config_path = 'cased_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'cased_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'cased_L-12_H-768_A-12/vocab.txt'

# token_dict = {}
# with codecs.open(dict_path, 'r', 'utf8') as reader:
#     for line in reader:
#         token = line.strip()
#         token_dict[token] = len(token_dict)

# '''
# 重写了Tokenizer的_tokenize方法，保证tokenize之后的结果，跟原来的字符串长度等长（如果算上两个标记，那么就是等长再加2）
# Tokenizer自带的_tokenize会自动去掉空格，导致tokenize之后的列表不等于原来字符串的长度了，这样如果做序列标注的任务会很麻烦。而为了避免这种麻烦，用[unused1]来表示空格类字符，而其余的不在列表的字符用[UNK]表示
# '''
# class OurTokenizer(Tokenizer):
#     def _tokenize(self, text):
#         R = []
#         text = text.split(' ')
#         for w in text:
#             if w in self._token_dict:
#                 R.append(w)
#             else:
#                 R.append('[UNK]') # 剩余的字符是[UNK]
#         return R

# tokenizer = OurTokenizer(token_dict)
# out = tokenizer.tokenize(u'a b c unaffable.')
# print(out)
# # 输出是 ['[CLS]', 'a', 'b', 'c', '[UNK]', '[SEP]']

# tokenizer = Tokenizer(token_dict)
# print(tokenizer.tokenize('a b c unaffable.'))  # The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]']`



# def seq_padding(X, padding=0):
#     L = [len(x) for x in X]
#     ML = max(L)
#     return np.array([
#         np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
#     ])


# class data_generator:
#     def __init__(self, data, batch_size=32):
#         self.data = data
#         self.batch_size = batch_size
#         self.steps = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1
#     def __len__(self):
#         return self.steps
#     def __iter__(self):
#         while True:
#             idxs = range(len(self.data))
#             np.random.shuffle(idxs)
#             X1, X2, Y = [], [], []
#             for i in idxs:
#                 d = self.data[i]
#                 text = d[0][:maxlen]
#                 x1, x2 = tokenizer.encode(first=text)
#                 y = d[1]
#                 X1.append(x1)
#                 X2.append(x2)
#                 Y.append([y])
#                 if len(X1) == self.batch_size or i == idxs[-1]:
#                     X1 = seq_padding(X1)
#                     X2 = seq_padding(X2)
#                     Y = seq_padding(Y)
#                     yield [X1, X2], Y
#                     [X1, X2, Y] = [], [], []



# bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path) # , output_layer_num=4, trainable=['Encoder']
# for l in bert_model.layers:
#     l.trainable = True
# bert_model.summary()


# x1_in = Input(shape=(None,))
# x2_in = Input(shape=(None,))

# x = bert_model([x1_in, x2_in])
# x = Lambda(lambda x: x[:, 0])(x)
# p = Dense(1, activation='sigmoid')(x)

# model = Model([x1_in, x2_in], p)
# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(1e-5), # 用足够小的学习率
#     metrics=['accuracy']
# )
# model.summary()


# train_D = data_generator(train_data)
# valid_D = data_generator(valid_data)

# model.fit_generator(
#     train_D.__iter__(),
#     steps_per_epoch=len(train_D),
#     epochs=5,
#     validation_data=valid_D.__iter__(),
#     validation_steps=len(valid_D)
# )






import keras.backend as K
import tensorflow_hub as hub
from utils.tokenization import FullTokenizer

# set GPU memory
if 'tensorflow'==K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    # # # 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # 按需求增长
    # sess = tf.Session(config=config)
    # set_session(sess)

    # 方法2:只允许使用x%的显存,其余的放着不动
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6    # 按比例
    sess = tf.Session(config=config)
    set_session(sess)


bert_path = 'bert/moduleB'
def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

tokens_a = tokenizer.tokenize('i like china')
print(tokens_a)