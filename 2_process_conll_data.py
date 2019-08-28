'''
语料预处理+特征抽取(语言学特征,词典特征,大小写,ELMO特征)
'''
import string
import pickle as pkl
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
from tqdm import tqdm
import word2vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import sys
sys.path.append("..")
from utils.helpers import wordNormalize, createCharDict
from utils.helpers import get_stop_dic
# import nltk
# nltk.download()
from nltk.corpus import stopwords



corpusPath = r'data'
embeddingPath = r'embedding'
embeddingFile = r'/home/administrator/PycharmProjects/embedding/pyysalo2013/wikipedia-pubmed-and-PMC-w2v.bin'
dict2idx = {'O': 0, 'B': 1, 'I': 2}  # 字典特征
label2idx = {'O': 0, 'B-protein': 1, 'I-protein': 2, 'B-gene': 3, 'I-gene': 4}

maxlen_s = 455  # 句子截断长度
maxlen_w = 21  # 单词截断长度
word_size = 200  # 词向量维度
MAX_NB_WORDS = None  # 不设置最大词数
word_len_list = [0]  # 用于统计单词长度
sen_len_list = [0]  # 用于统计句子长度



def readBinEmbedFile(embFile):
    """
    读取二进制格式保存的词向量文件
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    model = word2vec.load(embFile)
    print('加载词向量文件完成')
    for i in tqdm(range(len(model.vectors))):
        vector = model.vectors[i]
        word = model.vocab[i].lower()   # convert all characters to lowercase
        embeddings[word] = vector
    return embeddings


def readTxtEmbedFile(embFile):
    """
    读取预训练的词向量文件 
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    with open(embFile) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if len(line.split())<=2:
            continue
        values = line.strip().split()
        word = values[0].lower()
        vector = np.asarray(values[1:], dtype=np.float32)
        embeddings[word] = vector
    return embeddings


def readGensimFile(embFile):
    print("\nProcessing Embedding File...")
    import gensim
    model = gensim.models.Word2Vec.load(embFile)  # 'word2vec_words.model'
    word_vectors = model.wv
    return word_vectors


def produce_matrix(word_index, embedFile):
    '''
    生成词向量矩阵 embedding_matrix
    '''
    stopWord_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/stopwords_gene'  # 停用词词典
    stop_word = []
    with open(stopWord_path, 'r') as f:
        for line in f:
            stop_word.append(line.strip('\n'))
    stop_word.extend(stopwords.words('english'))
    stop_word.extend(list(string.punctuation))
    stop_word = list(set(stop_word))

    word_embeddings = readBinEmbedFile(embedFile)  # 读取词向量
    print('Found %s word vectors.' % len(word_embeddings))  # 4706287

    word_id_filter = []
    miss_num=0   # 未登陆词数量
    num=0   # 登陆词数量
    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words, word_size))
    for word, i in word_index.items():
        vec = None # 初始化为空
        if word in stop_word:
            word_id_filter.append(i)
        if word.lower() in word_embeddings:
            vec = word_embeddings.get(word.lower())
        else:
            for punc in string.punctuation:
                word = word.replace(punc, '')
            vec = word_embeddings.get(word.lower())
        if vec is not None:
            num=num+1
        else:
            miss_num=miss_num+1
            vec = word_embeddings["UNKNOWN_TOKEN"] # 未登录词均统一表示
        embedding_matrix[i] = vec

    print('未登陆词数量', miss_num) #3937
    print('登陆词数量', num) # 23019
    print('停用词id:{}'.format(list(set(word_id_filter))))
    return embedding_matrix


def padCharacters(chars_dic, max_char):
    '''
    补齐字符长度
    '''
    for senIdx in tqdm(range(len(chars_dic))):
        for tokenIdx in range(len(chars_dic[senIdx])):
            token = chars_dic[senIdx][tokenIdx]
            lenth = max_char - len(token)
            if lenth >= 0:
                chars_dic[senIdx][tokenIdx] = np.pad(token, (0, lenth), 'constant')
            else:
                chars_dic[senIdx][tokenIdx] = token[:max_char]
            assert len(chars_dic[senIdx][tokenIdx])==max_char
    return chars_dic


def getCasing(word):
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing
     

def getCasingVocab():
    entries = ['other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}


def getData(trainCorpus, sen_len_list):
    '''
    获取模型的训练数据和测试数据
    '''
    nb_word = 0     # 用于统计句子的单词个数
    chars_not_exit = set()     # 统计未登录字符
    char2idx = createCharDict()    # 字符字典
    casing_vocab = getCasingVocab()    # 大小写字典

    pos2idx = OrderedDict()
    pos2idx['None'] = 0
    chunk2idx = {'None': 0}

    datasDic = {'train':[], 'devel':[], 'test':[]}
    charsDic = {'train':[], 'devel':[], 'test':[]}
    capDic = {'train':[], 'devel':[], 'test':[]}
    posDic = {'train': [], 'devel': [], 'test':[]}
    chunkDic = {'train': [], 'devel': [], 'test':[]}
    labelsDic = {'train':[], 'devel':[], 'test':[]}
    dictDic = {'train':[], 'devel':[], 'test':[]}
    # ngramDic = {'train':[], 'devel':[], 'test':[]}

    len_list={'train':[], 'test':[]}
    for name in ['train', 'test']:
        # with open(trainCorpus + '/' + name + '_n_grams.txt', encoding='utf-8') as f:  # n元词典特征
        with open(trainCorpus + '/' + name + '.final.txt', encoding='utf-8') as f:  # 分布式词典特征
            data_sen = []
            char_sen = []
            cap_sen = []
            pos_sen = []
            chunk_sen = []
            labels_sen = []
            dict_sen = []
            # ngram_sen = []
            num = -1
            line_number = 0
            for line in f:
                if line == '\n':
                    len_list[name].append(nb_word)
                    line_number +=1
                    num += 1
                    if nb_word > sen_len_list[-1]:
                        sen_len_list.append(nb_word)
                    assert len(data_sen) == len(labels_sen)
                    if nb_word <= maxlen_s:
                        datasDic[name].append(' '.join(data_sen))  # .join()
                        charsDic[name].append(char_sen)
                        capDic[name].append(cap_sen)
                        posDic[name].append(pos_sen)
                        chunkDic[name].append(chunk_sen)
                        labelsDic[name].append(labels_sen)
                        dictDic[name].append(dict_sen)
                        # ngramDic[name].append(ngram_sen)
                    else:
                        datasDic[name].append(' '.join(data_sen[:maxlen_s]))  # .join()
                        charsDic[name].append(char_sen[:maxlen_s])
                        capDic[name].append(cap_sen[:maxlen_s])
                        posDic[name].append(pos_sen[:maxlen_s])
                        chunkDic[name].append(chunk_sen[:maxlen_s])
                        labelsDic[name].append(labels_sen[:maxlen_s])
                        dictDic[name].append(dict_sen[:maxlen_s])
                        # ngramDic[name].append(ngram_sen[:maxlen_s])

                    assert len(data_sen[:maxlen_s])==len(labels_sen[:maxlen_s])

                    data_sen = []
                    char_sen = []
                    cap_sen = []
                    pos_sen = []
                    chunk_sen = []
                    labels_sen = []
                    dict_sen = []
                    nb_word = 0
                else:
                    nb_word+=1
                    line_number += 1
                    token = line.replace('\n', '').split('\t')
                    word = token[0]
                    pos = token[1]
                    chunk = token[2]
                    dic = token[3]  # 分布式词典特征
                    label = token[4]
                    # ngram = list(map(int, token[5:]))  # n元词典特征

                    labelIdx = label2idx.get(label) if label in label2idx else label2idx.get('O')
                    labelIdx = np.eye(len(label2idx))[labelIdx]
                    labelIdx = list(labelIdx)

                    # 大小写特征
                    cap = casing_vocab[getCasing(word)]
                    # 对单词进行清洗
                    word = wordNormalize(word)
                    # 获取pos和chunk字典
                    if not pos in pos2idx:
                        pos2idx[pos] = len(pos2idx)
                    if not chunk in chunk2idx:
                        chunk2idx[chunk] = len(chunk2idx)
                    # 字符特征
                    nb_char = 0
                    char_w = []
                    for char in word:
                        nb_char+=1
                        charIdx = char2idx.get(char)
                        if not charIdx:
                            chars_not_exit.add(char)
                            char_w.append(char2idx['**'])
                        else:
                            char_w.append(charIdx)
                    if nb_char > word_len_list[-1]:
                        word_len_list.append(nb_char)
                    # 字典特征
                    dict_fea = dict2idx[dic]

                    data_sen.append(word)
                    char_sen.append(char_w)
                    cap_sen.append(cap)
                    pos_sen.append(pos)
                    chunk_sen.append(chunk)
                    labels_sen.append(labelIdx)
                    dict_sen.append(dict_fea)
                    # ngram_sen.append(ngram)

    print('chars not exits in the char2idx:{}'.format(list(chars_not_exit)))
    print('longest char is', word_len_list[-5:])  # [12, 14, 17, 21, 34]
    print('longest word is', sen_len_list[-5:])  # [370, 422, 455, 752, 902]
    print('len(pos2idx):{}'.format(len(pos2idx)))     # 50
    print('len(chunk2idx):{}'.format(len(chunk2idx)))     # 22
    a = sorted(len_list['train'])
    b = sorted(len_list['test'])
    print('longest word is : {}, {}'.format(a[-10:], b[-10:]))

    # return datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, ngramDic, pos2idx, chunk2idx
    return datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, pos2idx, chunk2idx


def main():
    '''
    主方法
    '''
    # stop_word_dic = get_stop_dic()
    # datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, ngramDic, pos2idx, chunk2idx = getData(corpusPath, sen_len_list)
    datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, pos2idx, chunk2idx = getData(corpusPath, sen_len_list)

    with open('pos2idx.txt', 'w') as f:
        for key, value in pos2idx.items():
            if key:
                f.write('{}\t{}\n'.format(key, value))
    with open('chunk2idx.txt', 'w') as f:
        for key, value in chunk2idx.items():
            if key:
                f.write('{}\t{}\n'.format(key, value))

    elmo_input = {}
    for name in ['train', 'test']:
        elmo_input[name] = []
        for i in range(len(datasDic[name])):
            line = datasDic[name][i]
            line = line.split()
            # line = text_to_word_sequence(line)
            elmo_input[name].append(line)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='',   # 需要过滤掉的字符列表
                          split=' ')    # 词的分隔符
    tokenizer.fit_on_texts(datasDic['train']+datasDic['test'])
    word_index = tokenizer.word_index   # 将词（字符串）映射到索引（整型）的字典
    word_counts = tokenizer.word_counts # 在训练时将词（字符串）映射到其出现次数的字典
    print('Found %s unique tokens.' % len(word_index))  # 26987

    with open('word_index.pkl', "wb") as f:
        pkl.dump(word_index, f, -1)

    # 数据序列化
    datasDic['train'] = tokenizer.texts_to_sequences(datasDic['train'])
    datasDic['test'] = tokenizer.texts_to_sequences(datasDic['test'])

    # 保证训练数据与标签长度一致
    for name in ['train', 'test']:
        for i in range(len(datasDic[name])):
            assert len(datasDic[name][i]) == len(labelsDic[name][i])== len(elmo_input[name][i])

    # pos特征序列化
    for name in ['train', 'test']:
        for i in range(len(posDic[name])):
            sent = posDic[name][i]
            posDic[name][i] = [pos2idx[item] for item in sent]

    # chunk特征序列化
    for name in ['train', 'test']:
        for i in range(len(chunkDic[name])):
            sent = chunkDic[name][i]
            chunkDic[name][i] = [chunk2idx[item] for item in sent]

    # 补齐字符长度
    charsDic['train'] = padCharacters(charsDic['train'], maxlen_w)
    charsDic['test'] = padCharacters(charsDic['test'], maxlen_w)

    # 获取词向量矩阵
    embedding_matrix = produce_matrix(word_index, embeddingFile)

    # 保存数据文件(XX.pkl or XX_ngram.pkl)
    with open(corpusPath+'/trainpkl', "wb") as f:
        pkl.dump((datasDic['train'], elmo_input['train'], labelsDic['train'], charsDic['train'],
                  capDic['train'], posDic['train'], chunkDic['train'], dictDic['train']), f, -1) # , ngramDic['train']
    with open(corpusPath+'/test.pkl', "wb") as f:
        pkl.dump((datasDic['test'], elmo_input['test'], labelsDic['test'], charsDic['test'],
                  capDic['test'], posDic['test'], chunkDic['test'], dictDic['test']), f, -1) # , ngramDic['test']

    with open(embeddingPath + '/emb.pkl', "wb") as f:
        pkl.dump((embedding_matrix), f, -1)
    with open(embeddingPath + '/length.pkl', "wb") as f:
        pkl.dump((maxlen_w, maxlen_s), f, -1)
    embedding_matrix = {}

    print('\n保存成功')


if __name__ == '__main__':
    main()
