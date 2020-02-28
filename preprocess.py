import pandas as pd
from sklearn.utils import shuffle
import sys
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import punkt
#stop_words = stopwords.words('english')
import numpy as np
import re, random
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.chunk import RegexpParser
import nltk, scipy, emoji
from nltk.corpus import wordnet
import csv, sys, random, math, re, itertools
from sklearn.utils import shuffle
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import enchant
d = enchant.Dict("en_US")

snowball_stemmer = SnowballStemmer('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer()


CONTRACTIONS = { 
"aint": " am not",
"ain't": "am not",
"aren't": "are not",
"arent": " are not",
"can't": "cannot",
"cant": " cannot",
"can't've": "cannot have",
"cant've": " cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldnt": " could not",
"couldn't've": "could not have",
"didn't": "did not",
"didnt": " did not",
"doesn't": "does not",
"doesnt": " does not",
"don't": "do not",
"dont": " do not",
"hadn't": "had not",
"hadnt": " had not",
"hadn't've": "had not have",
"hasn't": "has not",
"hasnt": "has not",
"haven't": "have not",
"havent": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"howll": "how will",
"how's": "how is",
"hows": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"Im": "I am",
"I've": "I have",
"Ive": "I have",
"isn't": "is not",
"isnt": "is not",
"it'd": "it would",
"itd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"itll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"lets": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightve": "might have",
"mightn't": "might not",
"mightnt": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustve": "must have",
"mustn't": "must not",
"mustnt": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"neednt": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"shes": "she is",
"should've": "should have",
"shouldve": "should have",
"shouldn't": "should not",
"shouldnt": "should not",
"shouldn't've": "should not have",
"shouldnt've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"thatd": "that had",
"that'd've": "that would have",
"that's": "that is",
"thats": "that is",
"there'd": "there would",
"thered": "there would",
"there'd've": "there would have",
"there's": "there is",
"theres": "there is",
"they'd": "they would",
"theyd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"theyll": "they will",
"they'll've": "they will have",
"they're": "they are",
"theyre": "they are",
"they've": "they have",
"theyve": "they have",
"to've": "to have",
"wasn't": "was not",
"wasnt": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weve": "we have",
"weren't": "were not",
"werent": "were not",
"what'll": "what will",
"whatll": "what will",
"what'll've": "what will have",
"what're": "what are",
"whatre": "what are",
"what's": "what is",
"whats": "what is",
"what've": "what have",
"whatve": "what have",
"when's": "when is",
"whens": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"wheres": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"wont": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldve": "would have",
"wouldn't": "would not",
"wouldnt": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"yall": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"youd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"youll": "you will",
"you'll've": " you will have",
"you're": "you are",
"youre": "you are",
"you've": "you have",
"youve": "you have"
}

def remove_adjacent_duplicates(word_list):
    curr = None
    new_word_list = []
    for i in range(len(word_list)):
        if curr is None:
            curr = word_list[i]
            new_word_list.append(curr)
            continue
        if word_list[i] != curr:
            curr = word_list[i]
            new_word_list.append(curr)
    return new_word_list

def remove_adjacent_duplicates_fromline(line):
    #word_list = nltk.word_tokenize(line.split()
    tknzr = TweetTokenizer()
    word_list = tknzr.tokenize(line)
    #new_word_list = [word for word in word_list if len(word) > 2]
    return ' '.join(remove_adjacent_duplicates(word_list))

def preprocess_1(sentence):

    if type(sentence) != str:
        return ""
    
    sentence = (sentence.encode('ascii', 'ignore')).decode("utf-8")
    
    # URLs
    sentence = re.sub(r'http\S+', ' <URL> ', sentence)
    
    # emoji
    for c in sentence:
        if c in emoji.UNICODE_EMOJI:
            sentence = re.sub(c, emoji.demojize(c), sentence)
    
    sentence = re.sub("([!]){1,}", " ! ", sentence)
    sentence = re.sub("([.]){1,}", " . ", sentence)
    sentence = re.sub("([?]){1,}", " ? ", sentence)
    sentence = re.sub("([;]){1,}", " ; ", sentence)
    sentence = re.sub("([:]){2,}", " : ", sentence)
    
    # numerical values
    #sentence = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " <NUMBER> ", sentence)
    
    # convert words such as "goood" to "good"
    sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))
    
    # convert to lower case
    words = tknzr.tokenize(sentence)
    
    # expand contractions
    words = [CONTRACTIONS[word.lower().strip()] if word.lower().strip() in CONTRACTIONS else word for word in words]
    
    new_words = []
    for word in words:
        if d.check(word.lower()):
            new_words.append(word)
            continue
        if len(word) < 2:
            new_words.append(word)
        else:
            if word[0] == 'n': # and word[1].isupper():
                new_words.append(word[1:])
            elif word[0].islower() and word[1].isupper() and d.check(word[1:]):
                new_words.append(word[1:])
            else:
                new_words.append(word)
    words = new_words

    f = open('slangs/slangs.txt')
    lines = f.readlines()
    f.close()
    slangs = {}
    for line in lines:
        slangs[line.split()[0]] = " ".join(line.split()[1:])

    f = open('slangs/slangs2.txt')
    lines = f.readlines()
    f.close()
    for line in lines:
        slangs[line.split()[0]] = " ".join(line.split()[1:])

    words = [slangs[word.lower().strip()] if word.lower().strip() in slangs else word for word in words]
 
    sentence =  " ".join(words)
    
    sentence = re.sub('[^ a-zA-Z0-9.!?:;<>_#@&]', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    
    return remove_adjacent_duplicates_fromline(sentence)

def preprocess_2(sentence): # after phrase thing
    
    if type(sentence) != str:
        return ""
    
    sentence = (sentence.encode('ascii', 'ignore')).decode("utf-8")
    
    # numerical values
    sentence = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", sentence)
    
    # convert to lower case
    words = tknzr.tokenize(sentence)

    '''
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
     "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
      'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
       'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
         'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
          'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
           'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and','s', 't', 'd', 'll',
             'm', 'o','re', 've', 'y', 'ma']
    '''

    #words = [word for word in words if word not in stop]
    words = ["<number>" if word.isdigit() else word for word in words]
    words = ["<@>" if '@' in word else word for word in words]
    
    special_words = ['.', '!', '?', ':', ';', '_', '-','<','>','#','@', '&', '<number>', '<@>', '<URL>']
    #words = [word.strip() for word in words if len(word) > 1 or word in special_words]

    f = open('slangs/some_keywords.txt')
    lines = f.readlines()
    f.close()
    some_keywords = set()
    for line in lines:
        some_keywords.add(line.strip().lower())

    eng_words = []
    for word in words:
        if word.lower() in some_keywords:
            eng_words.append(word)
            continue
        if d.check(word.lower()) or word in special_words or word[0] in special_words:
            eng_words.append(word)
        elif things2try(word) is not None:
            eng_words.append(things2try(word))
        elif word[0].isupper():
            eng_words.append(word)
        elif len(word) > 3:
            suggestions = d.suggest(word.lower())
            if len(suggestions) < 3 and len(suggestions) > 0:
                eng_words.append(suggestions[0]) 
            #else:
                #print(word)
        #else:
            #print(word)#, ' sugg-', d.suggest(word.lower()))
    
    sentence =  " ".join(eng_words)
    
    sentence = re.sub('[^ a-zA-Z0-9.!?:;<>_#@&]', ' ', sentence)
    sentence = re.sub('&', " and ", sentence)
    sentence = re.sub('\s+', ' ', sentence)
    sentence = sentence.lower()
    
    return remove_adjacent_duplicates_fromline(sentence)

def things2try(word):
    one = ''.join(''.join(s)[:1] for _, s in itertools.groupby(word.lower()))
    if d.check(one): # or in slangs etc
        #print(word," - ", one)
        return one
    if word[-1] == 's':
        if d.check(word[:-1]):
            #print(word," - ", word[:-1])
            return word[:-1]
    if word[-1] == 'n':
        if d.check(word+'g'):
            #print(word," - ", word+'g')
            return word+'g'
    return None

def full_preprocess(sentence):
    return preprocess_2(preprocess_1(sentence))

def contains_digits(word):
    for ch in word:
        if ch.isdigit():
            return True
    return False
    

