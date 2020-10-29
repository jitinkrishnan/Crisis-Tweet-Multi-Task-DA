# Unsupervised and Interpretable Domain Adaptation to Rapidly Filter Social Web Data for Emergency Services

## Aims to help emergency responders during Natural Disasters 

### Paper/Cite
https://arxiv.org/pdf/2003.04991.pdf (To appear at [ASONAM 2020](http://asonam.cpsc.ucalgary.ca/2020/))
```
@article{krishnanCrisisTweets,
  title={Unsupervised and Interpretable Domain Adaptationto Rapidly Filter Tweets for Emergency Services},
  author={Krishnan, Jitin and Purohit, Hemant and Rangwala, Huzefa},
  journal={In Proceedings of the 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  year={2020}
}
```

**Purpose of the model**: Train a classifier using past crisis-events where plenty of data is available and generalize it to a new crisis-event with **zero** data. 

### Why use this method?
- Unsupervised (no labeled target data is needed)
- Uses Multi-Task Learning to create a better generalizable model for the Low-Resource Crisis-Related Web Data (Tweets)
- Interpretable Predictions

### Requirements
Python3.6, Keras, Tensorflow.
Or ```pip install -r requirements.txt``` to install necessary packages.

### Additional Requirements
Download fastText [crawl-300d-2M-subword.bin](https://fasttext.cc/docs/en/english-vectors.html) to the current folder. For a smaller fastText word vector file with only words from 2018 TREC task, [click here](https://drive.google.com/open?id=1dNYCD5vuuGjT-BT-ZMfKYttU1UtUlOQg) to download from our google drive. If you are using the bin file, set ```binary=True``` in ```mtl.py```.

### TREC Data
[Click Here](https://github.com/jitinkrishnan/Crisis-Tweet-Multi-Task-DA/blob/master/TREC-MTL-DATASET-CONSTRUCTION.ipynb) to view the Jupyter Notebook that provides detailed instructions to construct the TREC datasets for MTL. We don't directly provide them because the dataset is not ours. 

### MTL: Sample Run
Run 
```
python mtl.py guatemalaEarthquake
``` 
This will train a MTL model by combining rest of the crisis events on the 4 tasks and will predict on *guatemalaEarthquake*.
Sample Result: 
```
[69.05, 84.05, 96.76, 93.11]
```
represents the acccuracy scores for the 4 tasks. 

### Visualize Attention: Sample Run
Format: ```python vizualize_attention.py <test-crisis> <filename> <task>```.
Run 
```
python vizualize_attention.py bostonBombings data/TASKS/priority/bostonBombings/pos priority
```
This will train a MTL model by combining rest of the crisis events on the 4 tasks. And then, it outputs the attention weights of the tweets from the given file for the specific task. 
Sample Result: 
```
tweet = "rt <@> <number> buff : boston : police and fbi urging anyone with video of the finish line at the time of the explosion should please come forward . ."
prediction = 1
attention = [('buff', 0.0254), ('boston', 0.0714), ('police', 0.0928), ('fbi', 0.0732), ('urging', 0.0831), ('anyone', 0.0324), ('with', 0.0359), ('video', 0.023), ('finish', 0.013), ('line', 0.0145), ('time', 0.0187), ('explosion', 0.0384), ('should', 0.0323), ('please', 0.0404), ('come', 0.0306), ('forward', 0.0379)]
```
[Click Here](https://github.com/jitinkrishnan/Crisis-Tweet-Multi-Task-DA/blob/master/visualize_attention.ipynb) to view the Jupyter Notebook that shows the attention heat map.

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).

