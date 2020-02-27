# Unsupervised Domain Adaptation for Multi-Task Classification of Low-Resource Crisis-Related Web Data

## Aims to help emergency responders during Natural Disasters 

**Purpose of the model**: Train a classifier using past crisis-events where plenty of data is available and generalize it to a new crisis-event with **zero** data. 

### Paper/Cite

### Why use this method? (See paper for detailed performance comparison)
- Unlike most state-of-the-art methods, no unlabeled target data is needed to train the model (which means no gradient reversal or manual pivot extractions). Out-of-the-box adaptable to any domain. 
- Computationally much cheaper than the state-of-the-art methods which uses unlabeled target data, with no trade-off in performance.

### Requirements
Python3.6, Keras, Tensorflow.
Or ```pip install -r requirements.txt``` to install necessary packages.

### Additional Requirements
Download [crawl-300d-2M-subword.bin](https://fasttext.cc/docs/en/english-vectors.html)

### TREC Data


### Sample Runs
#### BiLSTM
```python bilstm.py 'electronics' 'kitchen```


