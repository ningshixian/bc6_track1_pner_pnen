# keras_bc6_track1

**This repository** contains a BiLSTM-CRF implementation that used for protein/gene named entity recognition (PNER) and a attention-based implementation that used for protein/gene named entity normalization (PNEN). 

- The part of PNER is used to get the entity mentions. It integrates the Entity name knowledge and ELMo representations from the publication [Deep contextualized word representations](http://arxiv.org/abs/1802.05365) (Peters et al., 2018) into the [BiLSTM-CNN-CRF architecture](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/) and can improve the performance significantly. 

- The part of PNEN is used to generate candidate IDs and eliminate ambiguity for mentions. It integrates structural knowledge of entities into ID embeddings, which can be beneficial to remedy the entity ambiguity issue faced by PNEN. 

Trained models can be **stored** and **loaded** for inference. 
The implementation is based on Keras 2.2.0 and can be run with Tensorflow 1.8.0 as backend. It was optimized for Python 3.5 / 3.6. It does **not work** with Python 2.7.


**Requirements:**
* Python 3.6 - lower versions of Python do not work
* Keras 2.2.0 - For the creation of BiLSTM-CNN-CRF architecture
* Tensorflow 1.8.0 - As backend for Keras (other backends are untested.

**Note:** This implementation might be incompatible with different (e.g. more recent) versions of the frameworks. See [docker/requirements.txt](docker/requirements.txt) for a full list of all Python package requirements.


## Task introduction

**Bio-ID task**:
Bio-ID task is similar to the normalization tasks in previous BioCreative in that the goal is to link bioentities mentioned in the literature to standard database identifiers.

- Figure captions from full-length articles are provided.
- Multiple bioentities are annotated (Organisms and species, Genes and proteins, miRNA, smallmolecules, cellularcomponents, celltypes and cell lines, tissuesand organs).
- Teams can participate by annotating all or a subset of bioentities
- figure/panel number and associated text in BioC format

**Data sets**: 
The [training set](http://www.biocreative.org/resources/corpora/bcvi-bio-id-track/) consists of a collection of SourceData (3) annotated captions in BioC format. Each file contains all SourceData annotated captions for a given article for a total of 570 articles . 

**Evaluation**: 
The test data set is now available [here](http://www.biocreative.org/resources/corpora/bcvi-bio-id-track/). We will calculate precision, recall and F-measure. 
A scorer developed by MITRE will be used for the evaluation. Information about the scorer, and the scorer package can be found [here](http://www.biocreative.org/resources/corpora/bcvi-bio-id-track/). 



## Setup

In order to run the code, Python 3.6 or higher is required. The code is based on Keras 2.2.0 and as backend I recommend Tensorflow 1.8.0. I cannot ensure that the code works with different versions for Keras / Tensorflow or with different backends for Keras. The code **does not** work with Python 2.7.


### Installing the dependencies with pip

You can use `pip` to install the dependencies.

In [requirements.txt)](requirements.txt) you find a full list of all used packages. You can install it via:
```bash
pip install -r requirements.txt
```


# Get Started

- use `preprocessing` modules to get complex features;
- essential methods like `fit`, `score`, and `save`/`load` 

## preprocessing

```python
python 1_xml2conll_offset.py
python 1_xml2conll_offset.test.py
```

This script will first read the original xml file `xxx.xml`. The text will be then tagged with part of speech, chunking and tokenized using GENIA tagger tool. The tagged output will be written in a CoNLL format to standard out. Finally, you can obtain the processed data `train.out.txt` and `test.out.txt` .

```python
python 1_xml2dict.py
python 1_xml2dict.test.py
```

This script is used to extract dictionary features for the `xxx.out.txt` files obtained in the previous step, and to get the extracted `xxx.final.txt` .

```python
python 2_process_conll_data.py
```

This script will read the pretrained word embedding file `wikipedia-pubmed-and-PMC-w2v.bin`, generate word embedding matrix `embedding_matrix`, serialize the processed data, and extract the character features. Finally, it combines all the above features as the input format of the model and store a pickle file in the `data` folder. 


## Implementation with ELMo Word Representations

ELMo representations are computationally expensive to compute, but they usually improve the performance by about 1-5 percentage points F1-measure. If you want to use ELMo for better performance, you can download the reduced, pre-trained models from [here https://tfhub.dev/google/elmo/2](https://tfhub.dev/google/elmo/2?tf-hub-format=compressed)

The computation of ELMo representations is computationally expensive. A CNN is used to map the characters of a token to a dense vectors. These dense vectors are then fed through two BiLSTMs. The representation of each token and the two outputs of the BiLSTMs are used to form the final context-dependent word embedding.

```python
# Join the ELMO embedding
import tensorflow_hub as hub
elmo_model = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)
def ElmoEmbedding(x):
    elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), 
                        "sequence_len": tf.constant(batch_size*[sentence_maxlen]) },
                signature="tokens",
                as_dict=True)["elmo"]
elmo_emb= Lambda(ElmoEmbedding, output_shape=(sentence_maxlen, 1024))(elmo_input)
```


The `ElmoEmbedding`-class provides methods for the efficient computation of ELMo representations.


## PNER Training

See `3_nnet_trainer.py` for an example how to train PNER model. 

Main process:
1. Convert entity recognition tasks to sequence labeling problems;
2. Randomly initialize the extracted features;
3. Probabilistic prediction of input sequences using bidirectional LSTM;
4. Obtain the optimal labeling result by CRF+viterbi algorithm.

For training, you specify the datasets you want to train on:

```python
>>> with open('data/train.pkl', "rb") as f:
>>>     train_x, train_elmo, train_y, train_char, train_cap, train_pos, train_chunk, train_dict = pkl.load(f)
>>> with open('data/test.pkl', "rb") as f:
>>>     test_x, test_elmo, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)
```

After zero padding, you can then train the network in the following way:

```python
>>> model = buildModel()
>>> model.fit(x=dataSet['train'], y=train_y,
            epochs=15,
            batch_size=8,
            shuffle=True,
            callbacks=[calculatePRF1], 
            validation_split=0.2)
```

*Note:* ConllevalCallback Class will save the best model based on the performance of the model on the validation set after each iteration.



## PNEN Training

### 负采样

```
python ned/2_feature_extractor_train_old.py

正例: 抽取实体的上下文 以及 实体提及对应的正确ID
负例: 抽取实体的上下文 以及 实体提及对应的错误ID (设为5个)
```

See `ned_trainer_bmc.py` for an example how to train and evaluate this implementation. 

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


### Running a stored model

If enabled during the trainings process, models are stored to the 'models' folder. Those models can be loaded and be used to tag new data. 


### Evaluate your performance in one line:

See `_test_nnet.py` for an example how to evaluate PNER and PNEN model. 


```python
>>> model.score(x_test, y_test)
0.802  # f1-micro score
# For more performance, you have to use pre-trained word embeddings.
# For now, anaGo's best score is 90.94 f1-micro score.
```


## 评测脚本介绍

Each annotation is assigned to a label, according to the prefix of the value of its "type" infon.

[前缀：ID // 前缀：实体]
- Uniprot: (normalized), 
- protein: (unnormalized), 
- NCBI gene: (normalized), 
- gene: (unnormalized)

The scorer will report scores in **4** conditions: ( all annotations vs. normalized annotations only ) X ( strict span match vs. span overlap ). 
For each condition, mention-level recall/precision/fmeasure will be reported.

The scorer will also compute `recall`/`precision`/`fmeasure` on the normalized IDs which are found, both `micro-averaged` and
`macro-averaged`


## Documentation

(coming soon)

<!--
anaGo supports pre-trained word embeddings like [GloVe vectors](https://nlp.stanford.edu/projects/glove/).
-->

*Note:* If you have questions, feedback or find bugs, please send an email to me.


## Reference

This code is based on the following papers:

* Lample, Guillaume, et al. "[Neural architectures for named entity recognition.](https://arxiv.org/abs/1603.01360)" arXiv preprint arXiv:1603.01360 (2016).
* Peters, Matthew E., et al. "[Deep contextualized word representations.](https://arxiv.org/abs/1802.05365)" arXiv preprint arXiv:1802.05365 (2018).
* 《CNN-based ranking for biomedical entity normalization》
* 《ACL2018-Linking 》
* [Interactive Bio-ID Assignment Track (Bio-ID)](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-1/)
