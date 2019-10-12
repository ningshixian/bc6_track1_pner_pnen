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
* ...

**Note:** This implementation might be incompatible with different (e.g. more recent) versions of the frameworks. See [requirements.txt](requirements.txt) for a full list of all Python package requirements.


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

In [requirements.txt](requirements.txt), you can find a full list of all used packages. You can install it via:
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

This script will read the pretrained word embedding file `wikipedia-pubmed-and-PMC-w2v.bin` (you can download the word embedding file from [here](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin)), generate word embedding matrix `embedding_matrix`, serialize the processed data, and extract the character features. Finally, it combines all the above features as the input format of the model and store a pickle file in the **data** folder. 


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


The `ElmoEmbedding` class provides methods for the efficient computation of ELMo representations.


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

### Running the stored models

If enabled during the trainings process, models are stored to the 'results' folder. Those models can be loaded and be used to tag new data. Here, `ner_model/Model_Best.h5`.



## PNEN Training

### ID Representation Learning

KBs contain rich structural knowledge of entities (e.g., `name variations` and `entity ambiguity`), which can be formalized as constraints on embeddings and allows us to extend word embeddings to embeddings of entity ID. To this end, we adopt an `autoencoder` to learn the embedding of entity IDs based on Mention-Variant-ID structures provided by KBs.

With the help of the autoencoder, we can use the knowledge of entities from UniProt and NCBI Gene KBs to learn ID embeddings. Finally, its training results are saved in `embedding/synsetsVec.txt`.

See our [autoencoder project](https://github.com/ningshixian/AutoExtend) for more details. 


### Negative sampling

The purpose of this script is to generate positive and negative training examples for the entity disambiguation model.

- Positive example: Extract the context of the entity and the corresponding correct ID of the entity
- Negative example: Extract the context of the entity, but generate incorrect IDs for the entity by `candidate ID generation` module (maximum number is set to 5)

```python
python ned/1_feature_extractor_train_old.py
```

This script will use the APIs provided by Uniprot and NCBI Gene to generate candidate IDs for each entity, then read the pre-trained `ID embedding` learned by the `autoencoder`, and map all IDs to their corresponding vectors. Finally, you can obtain the processed data `data_train2.pkl`, `data_test2.pkl` and `id_embedding.pkl`.


### Entity disambiguation model Training

See `ned_trainer_bmc.py` for an example how to train entity disambiguation model. 

Main process:
1. The left context and the right context are first mapped through the word embedding matrix, and the ELMo representations are also added as in `PNER Training`;
2. After that, `Entity context representation learning` is performed via the `Semantic representation layer`. The `Semantic representation layer` includes: **CNN-based**, **LSTM-based**, **attention-based** and other sentence encoders;
3. Next is `Merge layer`, which includes two ways: gating mechanism and vector combination. It stitch [candidate ID, context] and input to softmax classification (0/1);
4. Finally, Calculate the similarity between all candidate IDs and the context, sort the scores of <m,c1>...<m,cx>. The candidate ID that gets the highest score will be taken as the final result of the mention.

For training, specifying the datasets:

```python
>>> with open('ned/data/data_train2.pkl', "rb") as f:
        x_left, x_pos_left, x_right, x_pos_right, y, x_elmo_l, x_elmo_r = pkl.load(f)
>>> with open('ned/data/id_embedding.pkl', "rb") as f:
        x_id, x_id_test, conceptEmbeddings = pkl.load(f)
```

Then train the network in the following way:

```python
>>> model = build_model()
    model.fit(x=dataSet, y=y,
              epochs=5,
              batch_size=32,
              shuffle=True,
              validation_split=0.2)
```

*Note:* the best model will be saved based on the performance of the model on the validation set after each iteration. Here, `ned/ned_model/weights_ned_max.hdf5` and `ned/ned_model/weights_ned_max.weights`. 



## Evaluate your performance:

See `_test_nnet.py` for an example how to evaluate PNER and PNEN model. 

```python
>>> dataSet = getTestData()
>>> model = load_model('ner_model/Model_Best.h5')
>>> predictions = model.predict(dataSet['test'])    # 预测
>>> y_pred = predictions.argmax(axis=-1)
>>> with open('result/predictions.pkl', "wb") as f:
        pkl.dump((y_pred), f, -1)
```

First, the test set is predicted by loading the previously trained **entity recognition model** to obtain entity mentions;

```python
>>> writeOutputToFile(test_file, y_pred, ned_model, prob, word_index, id2def, dict_or_text)
...
>>> result = searchEntityId(sentence, prediction, entity2id, text_byte, all_id)
>>> cnn = load_model('ned/ned_model/weights_ned_max.hdf5')
>>> type_id = entity_disambiguation_cnn(entity, entity_id, cnn, test_x[idx_line], sentence, test_pos[idx_line], x_id_dict, position, stop_word, leixing, prob, word_index, id2def, dict_or_text)
```

Then, by executing the **writeOutputToFile** function, the following operations are performed: (1) using the `knowledge base Retrieval (KB Retrieval)` to perform candidate ID generation on the entity mentions; (2) loading the trained **entity disambiguation model** to eliminate ambiguity for mentions.

```python
...
>>> outputName = result_path + '/' + file
>>> f = open(outputName, 'w')
>>> writer = codecs.lookup('utf-8')[3](f)
>>> dom.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
>>> writer.close()
>>> f.close()
...
>>> os.system('python /home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_score.py '
                '--verbose 1 --force '
                'ned/BioID_scorer_1_0_3/scripts/bioid_scores '
                'embedding/test_corpus_20170804/caption_bioc '
                'system1:embedding/test_corpus_20170804/prediction')
```

Finally, we write all the predictions in a new file in XML format and use the **official evaluation script** for evaluation.

Evaluation results can be viewed under the following path:
> ned/BioID_scorer_1_0_3/scripts/bioid_scores/corpus_scores.csv


## 评测脚本介绍

Each annotation is assigned to a label, according to the prefix of the value of its "type" infon.

- Uniprot: (normalized), 
- protein: (unnormalized), 
- NCBI gene: (normalized), 
- gene: (unnormalized)

The scorer will report scores in **4** conditions: ( **all annotations** vs. **normalized annotations only** ) X ( **strict span match** vs. **span overlap** ). 
For each condition, mention-level recall/precision/fmeasure will be reported.

The scorer will also compute `recall`/`precision`/`fmeasure` on the normalized IDs which are found, both `micro-averaged` and
`macro-averaged`


## Documentation

(coming soon)

*Note:* If you have questions, feedback or find bugs, please send an email to me.


## Data
We noticed that several factors could affect the replicatability of experiments:  
1. the segmentor for preprocessing: we used geniatagger-3.0.2   
2. the random number generator. Alghough we fixed the random seed, we noticed it will render slight different numbers on different machine.  
3. the traditional lexical feature used.  
4. the pre-trained embeddings.
To enhance the replicatability of our experiments, we provide the data we used in our experiments  in conll format (train/test).


## Reference

This code is based on the following papers:

* [Interactive Bio-ID Assignment Track (Bio-ID)](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-1/)
* Li H, Chen Q, et al. "[CNN-based ranking for biomedical entity normalization.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5629610/)" BMC bioinformatics. 2017;18(11):385.
* Eshel Y, Cohen N, et al. "[Named Entity Disambiguation for Noisy Text.](https://www.aclweb.org/anthology/K17-1008)" In: Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017). Vancouver, Canada. 2017:173-183.
* Luo L, Yang Z, et al. "[An attention-based BiLSTM-CRF approach to document-level chemical named entity recognition.]()" Bioinformatics. 2017;34(8):1381-1388.
* Kaewphan S, Hakala K, et al. "[Wide-scope biomedical named entity recognition and normalization with CRFs, fuzzy matching and character level modeling.]()" Database. 2018;2018(1):bay096.
