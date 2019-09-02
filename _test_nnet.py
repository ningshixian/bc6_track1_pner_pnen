import os
import pickle as pkl
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keraslayers.ChainCRF import create_custom_objects
from utils.write_test_result import writeOutputToFile
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

# # config = tf.ConfigProto()
# # # config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 按比例
# # # config.gpu_options.allow_growth = True  # 自适应分配
# # set_session(tf.Session(config=config))
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

'''
获取模型需要预测的测试数据
'''
def getTestData():

    with open('data/test.pkl', "rb") as f:
        test_x, test_elmo, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)

    embeddingPath = r'/embedding'
    with open(embeddingPath+'/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    dataSet = {}
    batch_size = 32
    sentence_maxlen = 400
    dataSet['test'] = [test_x, test_cap, test_pos, test_chunk, test_dict]


    # pad the sequences with zero
    for key, value in dataSet.items():
        for i in range(len(value)):
            dataSet[key][i] = pad_sequences(value[i], maxlen=sentence_maxlen, padding='post')

    # pad the char sequences with zero list
    for j in range(len(test_char)):
        test_char[j] = test_char[j][:sentence_maxlen]
        if len(test_char[j]) < sentence_maxlen:
            test_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(test_char[j]))))

    new_test_elmo = []
    for seq in test_elmo:
        new_seq = []
        for i in range(sentence_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_test_elmo.append(new_seq)
    test_elmo = np.array(new_test_elmo)

    dataSet['test'].insert(1, np.asarray(test_char))
    dataSet['test'].append(test_elmo)
    test_y = pad_sequences(test_y, maxlen=sentence_maxlen, padding='post')

    end2 = len(dataSet['test'][0]) // batch_size
    for i in range(len(dataSet['test'])):
        dataSet['test'][i] = dataSet['test'][i][:end2 * batch_size]

    dataSet['test'][1] = np.array(dataSet['test'][1])

    print(np.array(test_x).shape)  # (4528, )
    print(np.asarray(test_char).shape)     # (4528, 455, 21)
    print(test_y.shape)    # (4528, 455, 5)
    print('create test set done!\n')
    return dataSet


def main(ned_model=None, prob=5.5, word_index=None, id2def=None, ner_result='result/predictions.pkl'):
    # root = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/'
    # ner_result = 'result/predictions_bert.pkl'

    if not os.path.exists(ner_result):
        dataSet = getTestData()

        model = load_model('ner_model/Model_Best.h5')
        # model = load_model('model/Model_ST.h5', custom_objects=create_custom_objects())

        print('加载模型成功!!')

        predictions = model.predict(dataSet['test'])    # 预测
        y_pred = predictions.argmax(axis=-1)

        with open('result/predictions.pkl', "wb") as f:
            pkl.dump((y_pred), f, -1)
    else:
        # with open(root + 'result/predictions.pkl', "rb") as f:
        with open(ner_result, "rb") as f:
            y_pred = pkl.load(f)

    # 对实体预测结果y_pred进行链接，以特定格式写入XML文件
    test_file = r'data/test.final.txt'
    dict_or_text = 'id' # 'id' 'def'
    writeOutputToFile(test_file, y_pred, ned_model, prob, word_index, id2def, dict_or_text)

    '''
    利用scorer进行评估
    python bioid_score.py --verbose 1 --force \
    存放结果文件的目录 正确答案所在的目录 system1:预测结果所在的目录

    注意:不能有中文出现在命令中

    python /home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_score.py \
        --verbose 1 --force \
        ned/BioID_scorer_1_0_3/scripts/bioid_scores \
        embedding/test_corpus_20170804/caption_bioc \
        system1:embedding/test_corpus_20170804/prediction
    
    '''

if __name__ == '__main__':

    main(ner_result='result/predictions_bert_best.pkl')

    os.system('python /home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_score.py '
                '--verbose 1 --force '
                'ned/BioID_scorer_1_0_3/scripts/bioid_scores '
                'embedding/test_corpus_20170804/caption_bioc '
                'system1:embedding/test_corpus_20170804/prediction')

    res = []
    import csv
    csv_file = 'ned/BioID_scorer_1_0_3/scripts/bioid_scores/corpus_scores.csv'
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
        f.write('{}'.format(0))
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