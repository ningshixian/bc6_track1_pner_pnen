# keras_bc6_track1

**This repository** contains a BiLSTM-CRF implementation that used for protein/gene named entity recognition (PNER) and a attention-based implementation that used for protein/gene named entity normalization (PNEN). 

The part of PNER integrates the ELMo representations from the publication [Deep contextualized word representations](http://arxiv.org/abs/1802.05365) (Peters et al., 2018) into the [BiLSTM-CNN-CRF architecture](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/) and can improve the performance significantly for different sequence tagging tasks. The part of PNEN integrates

Trained models can be **stored** and **loaded** for inference. 

The implementation is based on Keras 2.2.0 and can be run with Tensorflow 1.8.0 as backend. It was optimized for Python 3.5 / 3.6. It does **not work** with Python 2.7.


**Requirements:**
* Python 3.6 - lower versions of Python do not work
* AllenNLP 0.5.1 - to compute the ELMo representations
* Keras 2.2.0 - For the creation of BiLSTM-CNN-CRF architecture
* Tensorflow 1.8.0 - As backend for Keras (other backends are untested.

**Note:** This implementation might be incompatible with different (e.g. more recent) versions of the frameworks. See [docker/requirements.txt](docker/requirements.txt) for a full list of all Python package requirements.


## 任务介绍

Bio-id任务: 进行7种类型（任选一种）的实体的识别和ID标注

- Organisms and species, 
- Genes and proteins, 
- miRNA, 
- smallmolecules,
- cellularcomponents, 
- celltypes and cell lines, 
- tissuesand organs. 


## Implementation with ELMo representations

The repository [bc6_track1_pner_pnen](https://github.com/ningshixian/bc6_track1_pner_pnen/edit/master/README.md) contains an extension of this architecture to work with the ELMo representations. ELMo representations are computationally expensive to compute, but they usually improve the performance by about 1-5 percentage points F1-measure. If you want to use ELMo for better performance(f1: 92.22), you can download the reduced, pre-trained models from here

In my experiments, I show that it is often sufficient to use only the first to layers of ELMo. The third layers led for various tasks to no significant improvement. Reducing the ELMo model from three to two layers increases the training speed up to 50%.


## Setup

In order to run the code, Python 3.6 or higher is required. The code is based on Keras 2.2.0 and as backend I recommend Tensorflow 1.8.0. I cannot ensure that the code works with different versions for Keras / Tensorflow or with different backends for Keras. The code **does not** work with Python 2.7.


### Installing the dependencies with pip

You can use `pip` to install the dependencies.

```bash
pip install allennlp==0.5.1 tensorflow==1.8.0 Keras==2.2.0
```

In [docker/requirements.txt)](docker/requirements.txt) you find a full list of all used packages. You can install it via:
```bash
pip install -r docker/requirements.txt
```


# Get Started

## preprocessing

An example is implemented in `xxx.py`:

```
python xxx.py 
```

This script will read the model `models/modelname.h5` as well as the text file `input.txt`. The text will be splitted into sentences and tokenized using NLTK. The tagged output will be written in a CoNLL format to standard out.


## Training

See `Train_Chunking.py` for an example how to train and evaluate this implementation. The code assumes a CoNLL formatted dataset like the CoNLL 2000 dataset for chunking.

For training, you specify the datasets you want to train on:

And you specify the pass to a pre-trained word embedding file:

```python
embeddingsPath = 'komninos_english_embeddings.gz'
```

The `util.preprocessing.py` fle contains some methods to read your dataset (from the `data` folder) and to store a pickle file in the `pkl` folder. 

You can then train the network in the following way:

```
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}
model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)
```


## ELMo Word Representations Computation

The `ELMoWordEmbeddings`-class provides methods for the efficient computation of ELMo representations. It has the following parameters:

The `ELMoWordEmbeddings` provides methods for the efficient computation of ELMo representations. It has the following parameters:
* `embeddings_file`: The ELMo paper concatenates traditional word embeddings, like GloVe, with the context dependent embeddings. With `embeddings_file` you can pass a path to a pre-trained word embeddings file. You can set it to `none` if you don't want to use traditional word embeddings.
* `elmo_options_file` and `elmo_weight_file`: AllenNLP provides different pretrained ELMo models.
* `elmo_mode`: Set to `average` if you want all 3 layers to be averaged. Set to `last` if you want to use only the final layer of the ELMo language model.
* `elmo_cuda_device`: Can be set to the ID of the GPU which should compute the ELMo embeddings. Set to `-1` to run ELMo on the CPU. Using a GPU drastically improves the computational time.


## Caching of ELMo representations

The computation of ELMo representations is computationally expensive. A CNN is used to map the characters of a token to a dense vectors. These dense vectors are then fed through two BiLSTMs. The representation of each token and the two outputs of the BiLSTMs are used to form the final context-dependent word embedding.


essential methods like `fit`, `score`, `analyze` and `save`/`load`. For more complex features, you should use the `models`, `preprocessing` modules and so on.

Here is the data loader:

```python
>>> from anago.utils import load_data_and_labels

>>> x_train, y_train = load_data_and_labels('train.txt')
>>> x_test, y_test = load_data_and_labels('test.txt')
>>> x_train[0]
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
>>> y_train[0]
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

You can now iterate on your training data in batches:

```python
>>> import anago

>>> model = anago.Sequence()
>>> model.fit(x_train, y_train, epochs=15)
Epoch 1/15
541/541 [==============================] - 166s 307ms/step - loss: 12.9774
...
```

Evaluate your performance in one line:

```python
>>> model.score(x_test, y_test)
0.802  # f1-micro score
# For more performance, you have to use pre-trained word embeddings.
# For now, anaGo's best score is 90.94 f1-micro score.
```


```python
# Transforming datasets.
p = ELMoTransformer()
p.fit(x_train, y_train)

# Building a model.
model = ELModel(...)
model, loss = model.build()
model.compile(loss=loss, optimizer='adam')

# Training the model.
trainer = Trainer(model, preprocessor=p)
trainer.train(x_train, y_train, x_test, y_test)
```


# Running a stored model

If enabled during the trainings process, models are stored to the 'models' folder. Those models can be loaded and be used to tag new data. 


## Documentation

(coming soon)

<!--
anaGo supports pre-trained word embeddings like [GloVe vectors](https://nlp.stanford.edu/projects/glove/).
-->


## 评测脚本介绍

Each annotation is assigned to a label, according to the prefix of the value of its "type" infon.

[前缀：ID // 前缀：实体]
- Uniprot: (normalized), 
- protein: (unnormalized), 
- NCBI gene: (normalized), 
- gene: (unnormalized)

The scorer will report scores in 4 conditions: ( all annotations vs. normalized annotations only ) X ( strict span match vs. span overlap ). For each condition, mention-level recall/precision/fmeasure will be reported.

The scorer will also compute recall/precision/fmeasure on the normalized IDs which are found, both micro-averaged and
macro-averaged



*Note:* If you have questions, feedback or find bugs, please send an email to me.


## Reference

This code is based on the following papers:

* Lample, Guillaume, et al. "[Neural architectures for named entity recognition.](https://arxiv.org/abs/1603.01360)" arXiv preprint arXiv:1603.01360 (2016).
* Peters, Matthew E., et al. "[Deep contextualized word representations.](https://arxiv.org/abs/1802.05365)" arXiv preprint arXiv:1802.05365 (2018).
