import numpy as np
from logging import info
from datetime import datetime
from abc import ABCMeta, abstractmethod
from keras.callbacks import Callback
from decimal import Decimal
import sys
sys.path.append("..")
from evaluation import conlleval
from evaluation import BIOF1Validation
import pickle as pkl


tag2id = {'O': 0, 'B-protein': 1, 'I-protein': 2, 'B-gene': 3, 'I-gene': 4}
tag = tag2id.keys()


class ConllevalCallback(Callback):
    '''
    Callback for running the conlleval script on the test dataset after each epoch.
    '''
    def __init__(self, X_test, y_test, samples, idx2label, sentence_maxlen, max_f, batch_size):
        super(ConllevalCallback, self).__init__()
        self.X = X_test
        self.y = np.array(y_test)
        self.samples = samples
        self.idx2label = idx2label
        self.sentence_maxlen = sentence_maxlen
        self.max_f = max_f
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        
        # if epoch<2:
        #     return

        print('开始测试')
        predictions = self.model.predict(self.X, batch_size=self.batch_size, verbose=1)
        y_pred = predictions.argmax(axis=-1)
        print(y_pred.shape) # (4528, 427)
        with open('result/predictions.pkl', "wb") as f:
            pkl.dump((y_pred), f, -1)

        #　脚本测试
        import csv
        import _test_nnet as m
        import os
        m.main(prob=1, ner_result='result/predictions.pkl') # predictions_76.31.pkl

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
        # if F>self.max_match:
        #     self.model.save('ned/ned_model/weights_ned_max.hdf5', overwrite=True)
        #     self.model.save_weights('ned/ned_model/weights_ned_max.weights', overwrite=True)
        #     self.max_match = F


class ConllevalCallback2(Callback):
    '''
    Callback for running the conlleval script on the test dataset after each epoch.
    '''

    def __init__(self, X_test, y_test, samples, idx2label, sentence_maxlen, max_f, batch_size):
        super(ConllevalCallback, self).__init__()
        self.X = X_test
        self.y = np.array(y_test)
        self.samples = samples
        self.idx2label = idx2label
        self.sentence_maxlen = sentence_maxlen
        self.max_f = max_f
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        print('开始测试')
        if self.samples:
            predictions = self.model.predict_generator(self.X, self.samples)
        else:
            predictions = self.model.predict(self.X, verbose=1)  # 模型预测

        # print(predictions[:2])
        y_pred = predictions.argmax(axis=-1)  # Predict classes [0]
        y_test = self.y.argmax(axis=-1)

        prf_file = 'prf.txt'
        # target = r'data/BC4CHEMD-IOBES/test.tsv'

        pre, rec, f1 = self.predictLabels2(y_pred, y_test)
        # p, r, f, c = self.predictLabels1(target, y_pred)

        if f1 >= self.max_f:
            self.max_f = f1
            self.model.save('Model_Best.h5', overwrite=True)
            self.model.save_weights('Model_weight_Best.h5', overwrite=True)
            # # 预测
            # model = load_model('model/Model_ST.h5', custom_objects=create_custom_objects())

        with open(prf_file, 'a') as pf:
            print('write prf...... ')
            pf.write("epoch= " + str(epoch + 1) + '\n')
            pf.write("precision= " + str(pre) + '\n')
            pf.write("recall= " + str(rec) + '\n')
            pf.write("Fscore= " + str(f1) + '\n')

    def predictLabels1(self, target, y_pred):
        s = []
        sentences = []
        s_num = 0
        with open(target) as f:
            for line in f:
                if not line == '\n':
                    s.append(line.strip('\n'))
                    continue
                else:
                    # if flag=='main' and not s_num<cal_batch(test_x):
                    #     break
                    # if flag=='aux' and not s_num<cal_batch(cdr_test_x):
                    #     break
                    prediction = y_pred[s_num]
                    s_num += 1
                    for i in range(len(s)):
                        if i >= self.sentence_maxlen: break
                        r = s[i] + '\t' + self.idx2label[prediction[i]] + '\n'
                        sentences.append(r)
                    sentences.append('\n')
                    s = []
        with open('../result/result.txt', 'w') as f:
            for line in sentences:
                f.write(str(line))

        p, r, f, c = conlleval.main((None, r'../result/result.txt'))
        return round(Decimal(p), 2), round(Decimal(r), 2), round(Decimal(f), 2), c


    def predictLabels2(self, y_pred, y_true):
        # y_true = np.squeeze(y_true, -1)
        lable_pred = list(y_pred)
        lable_true = list(y_true)

        print('\n计算PRF...')
        pre, rec, f1 = BIOF1Validation.compute_f1(lable_pred, lable_true, self.idx2label, 'O', 'BIO')
        print('precision: {:.2f}%'.format(100. * pre))
        print('recall: {:.2f}%'.format(100. * rec))
        print('f1: {:.2f}%'.format(100. * f1))

        return round(Decimal(100. * pre), 2), round(Decimal(100. * rec), 2), round(Decimal(100. * f1), 2)


class LtlCallback(Callback):
    """Adds after_epoch_end() to Callback.

    after_epoch_end() is invoked after all calls to on_epoch_end() and
    is intended to work around the fixed callback ordering in Keras,
    which can cause output from callbacks to mess up the progress bar
    (related: https://github.com/fchollet/keras/issues/2521).
    """

    def __init__(self):
        super(LtlCallback, self).__init__()
        self.epoch = 0

    def after_epoch_end(self, epoch):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0:
            self.after_epoch_end(self.epoch)
        self.epoch += 1

    def on_train_end(self, logs={}):
        self.after_epoch_end(self.epoch)

class CallbackChain(Callback):
    """Chain of callbacks."""

    def __init__(self, callbacks):
        super(CallbackChain, self).__init__()
        self._callbacks = callbacks

    def _set_params(self, params):
        for callback in self._callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self._callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_end(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_end(*args, **kwargs)

class EvaluatorCallback(LtlCallback):
    """Abstract base class for evaluator callbacks."""

    __metaclass__ = ABCMeta

    def __init__(self, dataset, label=None, writer=None):
        super(EvaluatorCallback, self).__init__()
        if label is None:
            label = dataset.name
        if writer is None:
            writer = info
        self.dataset = dataset
        self.label = label
        self.writer = writer
        self.summaries = []

    def __call__(self):
        """Execute Callback. Invoked after end of each epoch."""
        summary = self.evaluation_summary()
        self.summaries.append(summary)
        epoch = len(self.summaries)
        for s in summary.split('\n'):
            self.writer('{} Ep: {} {}'.format(self.label, epoch, s))

    @abstractmethod
    def evaluation_summary(self):
        """Return string summarizing evaluation results."""
        pass

    def after_epoch_end(self, epoch):
        self()

class EpochTimer(LtlCallback):
    """Callback that logs timing information."""

    def __init__(self, label='', writer=info):
        super(EpochTimer, self).__init__()
        self.label = '' if not label else label + ' '
        self.writer = writer

    def on_epoch_begin(self, epoch, logs={}):
        super(EpochTimer, self).on_epoch_begin(epoch, logs)
        self.start_time = datetime.now()

    def after_epoch_end(self, epoch):
        end_time = datetime.now()
        delta = end_time - self.start_time
        start = str(self.start_time).split('.')[0]
        end = str(end_time).split('.')[0]
        self.writer('{}Ep: {} {}s (start {}, end {})'.format(
                self.label, epoch, delta.seconds, start, end
                ))

class Predictor(LtlCallback):
    """Makes and stores predictions for data item sequence."""

    def __init__(self, dataitems):
        super(Predictor, self).__init__()
        self.dataitems = dataitems

    def after_epoch_end(self, epoch):
        predictions = self.model.predict(self.dataitems.inputs)
        self.dataitems.set_predictions(predictions)


class PredictionMapper(LtlCallback):
    """Maps predictions to strings for data item sequence."""

    def __init__(self, dataitems, mapper):
        super(PredictionMapper, self).__init__()
        self.dataitems = dataitems
        self.mapper = mapper

    def after_epoch_end(self, epoch):
        self.dataitems.map_predictions(self.mapper)
        # TODO check if summary() is defined
        info(self.mapper.summary())


class TokenAccuracyEvaluator(EvaluatorCallback):
    """Evaluates performance using token-level accuracy."""

    # TODO why does this class exist? Isn't TokenLevelEvaluator better
    # in every way?

    def __init__(self, dataset, label=None, writer=None):
        super(TokenAccuracyEvaluator, self).__init__(dataset, label, writer)

    def evaluation_summary(self):
        gold = self.dataset.tokens.target_strs
        pred = self.dataset.tokens.prediction_strs
        assert len(gold) == len(pred)
        total = len(gold)
        correct = sum(int(p==g) for p, g in zip(pred, gold))
        return 'acc: {:.2%} ({}/{})'.format(1.*correct/total, correct, total)


# if __name__ == '__main__':
#     path = '../model/Model_{}_{}.h5'.format(0, 81.29)
#     from keras.models import Model
#     from keras.layers import Input, Dense
#     input = Input(shape=(1,2))
#     output = Dense(2)(input)
#     model = Model(inputs=input, outputs=output)
#     model.save(path, overwrite=True)