# Unsupervised Domain Adaptation for Multi-Task Classification of Low-Resource Crisis-Related Web Data

## Aims to help emergency responders during Natural Disasters 

**Purpose of the model**: Train a classifier using past crisis-events where plenty of data is available and generalize it to a new crisis-event with **zero** data. 

### Why use this method?
- Unsupervised (no labeled target data is needed)
- Uses Multi-Task Learning to create a better generalized model for the Low-Resource dataset
- Interpretable Predictions

### Requirements
Python3.6, Keras, Tensorflow.
Or ```pip install -r requirements.txt``` to install necessary packages.

### Additional Requirements
Download fastText [crawl-300d-2M-subword.bin](https://fasttext.cc/docs/en/english-vectors.html) to the current folder. For a smaller fastText word vector file with only words from 2018 TREC task, [click here](https://drive.google.com/open?id=1dNYCD5vuuGjT-BT-ZMfKYttU1UtUlOQg). If you use the bin file, set ```binary=True``` in ```mtl.py```.

### TREC Data
[Click Here](https://github.com/jitinkrishnan/Crisis-Tweet-Multi-Task-DA/blob/master/TREC-MTL-DATASET-CONSTRUCTION.ipynb) to view the Jupyter Notebook that provides detailed instructions to construct the TREC datasets for MTL. We don't directly provide them because the dataset is not ours. 

### MTL Sample Run
Run ```python mtl.py guatemalaEarthquake```. This will train a MTL model by combining rest of the crisis events on the 4 tasks and will predict on *guatemalaEarthquake*.
Sample Result: ```[69.05, 84.05, 96.76, 93.11]``` represents the acccuracy scores for the 4 tasks. 

### Visualize Attention

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).

