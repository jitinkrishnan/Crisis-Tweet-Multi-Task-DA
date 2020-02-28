import numpy as np
from gensim.models.keyedvectors import KeyedVectors

def wordvec_dict(vec_file, binary=False):
	model = KeyedVectors.load_word2vec_format(vec_file,binary=binary)
	return model.wv

def lenOfdict(wv_dict):
    return len(wv_dict.vocab.keys())

def get_wordvector(word, model_wv,dim):
	vocab_keys = model_wv.vocab.keys()
	if word in vocab_keys:
		return model_wv[word]
	else:
		print("Not in: ", word)
	return None
