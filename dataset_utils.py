from preprocess import *
from wordvec_utils import *
import numpy as np
import nltk, random
from sklearn.metrics.pairwise import cosine_similarity
import random, copy

def create_one_training_example(full_text_example, max_len, wv_dict, dim=300):
    text = full_text_example.lower()
    words = text.split()
    bag = []
    mywords = []
    count = 0
    for word in words:
        if count == max_len:
            break
        if word in wv_dict.vocab.keys():
            v = get_wordvector(word,wv_dict,dim)
            if v is not None:
                count += 1
                bag.append(list(v))
                mywords.append(word)
    
    for i in range(max_len-count):
        bag.append(list(np.zeros(dim)))

    return mywords, np.asarray(bag)

def inplace_shuffle(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a,b

def get_vocab_dict(fname):
	f = open(fname)
	lines = f.readlines()
	f.close()
	vocab_dict = {}
	for line in lines:
		words = line.split()
		for word in words:
			try:
				vocab_dict[word] += 1
			except:
				vocab_dict[word] = 1
	return vocab_dict

def custom_key(item):
	return item[1]

def create_data4lstm_allvsone_mtl(task, train_category, test_category, wv_dict, Tx=30, Ty=1, dim=300,min_count=2750):

	# TRAIN
	pos = []
	neg = []
	for category in train_category:
		f_bags_pos = open("data/TASKS/"+task+"/"+category+"/pos")
		f_bags_neg = open("data/TASKS/"+task+"/"+category+"/neg")

		one_pos = f_bags_pos.readlines()
		one_neg = f_bags_neg.readlines()
		#bags = pos + neg

		f_bags_pos.close()
		f_bags_neg.close()

		pos.extend(one_pos)
		neg.extend(one_neg)

	random.shuffle(pos)
	random.shuffle(neg)

	pos_orig_copy = copy.deepcopy(pos)
	neg_orig_copy = copy.deepcopy(neg)

	while len(pos) < min_count:
		random.shuffle(pos_orig_copy)
		extra = copy.deepcopy(pos_orig_copy[:(min_count-len(pos))])
		pos.extend(extra)
	while len(neg) < min_count:
		random.shuffle(neg_orig_copy)
		extra = copy.deepcopy(neg_orig_copy[:(min_count-len(neg))])
		neg.extend(extra)

	min_num = min(len(pos), len(neg))

	bag_pos = []
	for text in pos[:min_num]:
		bag_pos.append(create_one_training_example(text, Tx, wv_dict, dim=dim)[1])

	bag_neg = []
	for text in neg[:min_num]:
		bag_neg.append(create_one_training_example(text, Tx, wv_dict, dim=dim)[1])

	pos_labels = []
	for i in range(len(bag_pos)):
		pos_labels.append([1,0])

	neg_labels = []
	for i in range(len(bag_neg)):
		neg_labels.append([0,1])

	X_train = bag_pos + bag_neg
	Y_train = pos_labels + neg_labels
	if len(X_train) > 0:
		(X_train,Y_train) = inplace_shuffle(X_train,Y_train)

	Xoh = np.asarray(X_train)
	Yoh = np.asarray(Y_train)

	Yoh = np.reshape(Yoh, (Yoh.shape[0],1,2))

	# TEST
	f_bags_pos = open("data/TASKS/"+task+"/"+test_category+"/pos")
	f_bags_neg = open("data/TASKS/"+task+"/"+test_category+"/neg")

	pos = f_bags_pos.readlines()
	neg = f_bags_neg.readlines()
	bags = pos + neg

	f_bags_pos.close()
	f_bags_neg.close()

	min_num = max(len(pos), len(neg)) # take all

	bag_pos = []
	for text in pos[:min_num]:
		bag_pos.append(create_one_training_example(text, Tx, wv_dict, dim=dim)[1])

	bag_neg = []
	for text in neg[:min_num]:
		bag_neg.append(create_one_training_example(text, Tx, wv_dict, dim=dim)[1])
	
	pos_labels = []
	for i in range(len(bag_pos)):
		pos_labels.append([1,0])

	neg_labels = []
	for i in range(len(bag_neg)):
		neg_labels.append([0,1])

	X_test = bag_pos + bag_neg
	Y_test = pos_labels + neg_labels
	if len(X_test) > 0:
		(X_test,Y_test) = inplace_shuffle(X_test,Y_test)

	Xoh_test = np.asarray(X_test)
	Yoh_test = np.asarray(Y_test)

	return Xoh, Yoh, Xoh_test, Yoh_test

def create_data4lstm_DA_mtl(train_category, wv_dict, Tx=75, Ty=1, dim=300, min_count=650):

    # TRAIN
    bag_label = []
    for index in range(len(train_category)):

        category = train_category[index]

        f_bags = open("data/domains_unlabeled/"+category)
        lines = f_bags.readlines()
        f_bags.close()

        bags = []
        labels = []
        for text in lines:
            bags.append(create_one_training_example(text, Tx, wv_dict, dim=dim)[1])
            label = [0]*len(train_category)
            label[index] = 1
            labels.append(label)

        random.shuffle(bags)

        if min_count > len(bags):
            extra = copy.deepcopy(bags[:(min_count-len(bags))])
            bags.extend(extra)
            for i in range(min_count-len(bags)):
                label = [0]*len(train_category)
                label[index] = 1
                labels.append(label)

        bag_label.append((bags,labels))

    x = [len(y[0]) for y in bag_label]
    min_num = min(x)

    ov_bags = []
    ov_labels = []
    for bl in bag_label:
        bags,labels = bl
        ov_bags.extend(bags[:min_num])
        ov_labels.extend(labels[:min_num])

    (ov_bags,ov_labels) = inplace_shuffle(ov_bags,ov_labels)

    X_train = ov_bags[:int(len(ov_bags)*0.85)]
    Y_train = ov_labels[:int(len(ov_labels)*0.85)]

    X_test = ov_bags[int(len(ov_bags)*0.85):]
    Y_test = ov_labels[int(len(ov_labels)*0.85):]

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    Y_train = np.reshape(Y_train, (Y_train.shape[0],1,len(train_category)))

    return X_train, Y_train, X_test, Y_test
