# Unsupervised Domain Adaptation for Multi-Task Classification of Low-Resource Crisis-Related Web Data

## Aims to help emergency responders during Natural Disasters 

**Purpose of the model**: Train a classifier using past crisis-events where plenty of data is available and generalize it to a new crisis-event with **zero** data. 

### Paper/Cite

### Why use this method? (See paper for detailed performance comparison)
- Unsupervised (no labeled target data is needed)
- Use Multi-Task Learning to create a better generalized model for the Low-Resource dataset
- Interpretable Predictions

### Requirements
Python3.6, Keras, Tensorflow.
Or ```pip install -r requirements.txt``` to install necessary packages.

### Additional Requirements
Download [crawl-300d-2M-subword.bin](https://fasttext.cc/docs/en/english-vectors.html)

### TREC Data
[Click Here](https://github.com/jitinkrishnan/Crisis-Tweet-Multi-Task-DA/blob/master/TREC-MTL-DATASET-CONSTRUCTION.ipynb) to view the Jupyter Notebook that provides detailed instructions to construct the TREC datasets for MTL.


### Sample Runs
#### BiLSTM
```python bilstm.py 'electronics' 'kitchen```


