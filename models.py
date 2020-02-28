import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import nltk, scipy, random
import pandas as pd
import sys, random, math, re, itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk import word_tokenize
import operator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from numpy import linalg as LA
import math
from dataset_utils import *
from sklearn.metrics import multilabel_confusion_matrix
from Gradient_Reverse_Layer import GradientReversal
import math, sys
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, GlobalMaxPooling1D

######################## helper functions ########################
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

############################### HELPER FUNCTION - GET ATTENTION WEIGHTS ################

def get_attention_weights(model, sentence, wv_dict, Tx, Ty, n_s=64, max_len=300):
    words, bag = create_one_training_example(sentence, Tx, wv_dict)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    prediction = model.predict([np.expand_dims(bag, axis=0), s0, c0])
    word_att = prediction[1].squeeze()
    y = np.argmax(prediction[0].squeeze())
    #print("ATT LEN: ", prediction[1].squeeze().shape)
    word_att = list(word_att)
    ans = []
    for index in range(min(len(words),max_len)):
        ans.append((words[index],round(word_att[index],4)))
    return y, ans

def get_attention_weights_mtl(model, sentence, wv_dict, Tx, Ty, n_s=64, max_len=300,task_num=1):
    words, bag = create_one_training_example(sentence, Tx, wv_dict,dim=max_len)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    prediction = model.predict([np.expand_dims(bag, axis=0),np.expand_dims(bag, axis=0),np.expand_dims(bag, axis=0),np.expand_dims(bag, axis=0),np.expand_dims(bag, axis=0), s0, c0])
    word_att = prediction[task_num+5].squeeze()
    y = np.argmax(prediction[task_num].squeeze())
    #print("ATT LEN: ", prediction[1].squeeze().shape)
    word_att = list(word_att)
    ans = []
    for index in range(min(len(words),max_len)):
        ans.append((words[index],round(word_att[index],4)))
    return y, ans

def get_attention_weights_nmtl(model, sentence, wv_dict, Tx, Ty, n_s=64, max_len=300,total_tasks=7,task_num=1):
    words, bag = create_one_training_example(sentence, Tx, wv_dict,dim=max_len)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    X = [np.expand_dims(bag, axis=0)]*(total_tasks+1)
    X.extend([s0, c0])
    prediction = model.predict(X)
    word_att = prediction[task_num+total_tasks+1].squeeze()
    y = np.argmax(prediction[task_num].squeeze())
    #print("ATT LEN: ", prediction[1].squeeze().shape)
    word_att = list(word_att)
    ans = []
    for index in range(min(len(words),max_len)):
        ans.append((words[index],round(word_att[index],4)))
    return y, ans

################################### BILSTM + ATTENTION + JOINT LEARNING + GRL #############################################################

def bilstm_attn_mtl(Tx, Ty, n_a, n_s, vocab_size, out_dim, dc_out_dim, drop=0.4,grl_lambda=0.4):

    X1 = Input(shape=(Tx, vocab_size))
    X2 = Input(shape=(Tx, vocab_size))
    X3 = Input(shape=(Tx, vocab_size))
    X4 = Input(shape=(Tx, vocab_size))
    X5 = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    common = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))

    a1 = common(X1)
    a2 = common(X2)
    a3 = common(X3)
    a4 = common(X4)
    a5 = common(X5)

    inputs = [a1,a2,a3,a4]
    outputs = []
    attentions = []

    count = 0

    for a in inputs:

        attn_weights = []

        s = s0
        c = c0

        for h in range(5):
            repeator = RepeatVector(Tx)
            concatenator = Concatenate(axis=-1)
            densor1 = Dense(10, activation = "tanh")
            densor2 = Dense(1, activation = "relu")
            activator = Activation(softmax)
            dotor = Dot(axes = 1)
            post_activation_LSTM_cell = LSTM(n_s, return_state = True)

            s_prev = repeator(s)
            concat = concatenator([a,s_prev])
            e = densor1(concat)
            e = Dropout(drop)(e)
            energies = densor2(e)
            alphas = activator(energies)
            #context = dotor([alphas,a])
            #s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
            attn_weights.append(alphas)

        alphas = keras.layers.Add()(attn_weights)
        context = dotor([alphas,a])
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])

        out_pre = Dense(n_s, activation = "relu")(s)
        output_layer = Dense(out_dim, activation=softmax, name='classification_'+str(count))(out_pre)

        word_attention = Lambda(lambda x: x[:, :,0,])(alphas)
        word_attention = Activation(None, name='word_attention_'+str(count))(word_attention)

        outputs.append(output_layer)
        attentions.append(word_attention)

        count += 1

    #outputs.extend(attentions)

    s = s0
    c = c0

    s_prev = repeator(s)
    concat = concatenator([a5,s_prev])

    e = Dense(10, activation = "tanh")(concat)
    e = Dropout(drop)(e)
    energies = Dense(1, activation = "relu")(e)
    alphas2 = Activation(softmax)(energies)
    context = Dot(axes = 1)([alphas2,a5])
    s, _, c = LSTM(n_s, return_state = True)(context, initial_state = [s,c])

    flip_layer = GradientReversal(grl_lambda)
    dann_in = flip_layer(s)
    inter_output_layer = Dense(n_s, activation = "relu")(dann_in)
    domain_classification = Dense(dc_out_dim, activation=softmax, name='domain_classification')(inter_output_layer)

    outputs.append(domain_classification)

    outputs.extend(attentions)

    model = Model(inputs=[X1, X2, X3, X4, X5, s0, c0], outputs=outputs)
    return model

def upbalance_dataset(X,Y,dc_dim):
    max_num = 0
    for el in X:
        if len(el) > max_num:
            max_num = len(el)

    new_X = []
    new_Y = []
    for index in range(len(X)):
        rem = max_num - len(X[index])
        if rem > 0:
            X_i, Y_i = inplace_shuffle(X[index], Y[index])
            new_X.append(X[index] + X_i[:rem])
            new_Y.append(Y[index] + Y_i[:rem])
        else:
            new_X.append(X[index])
            new_Y.append(Y[index])

    for el in new_X:
        print(len(el))
        print(max_num)
        #assert(max_num==len(el))

    for index in range(len(new_X)):
        new_X[index] = np.asarray(new_X[index])

    for index in range(len(new_Y)):
        if index == len(new_Y)-1:
            new_Y[index] = np.asarray(new_Y[index])
            new_Y[index] = np.reshape(new_Y[index], (new_Y[index].shape[0],1,dc_dim))
        else:
            new_Y[index] = np.asarray(new_Y[index])
            new_Y[index] = np.reshape(new_Y[index], (new_Y[index].shape[0],1,2))

    return new_X, new_Y

def evaluate_bilstm_attention_mtl(model, X, Y, n_s=64):

    accs = []
    f1s = []

    for index in range(len(X)):
        Xoh_test = X[index]
        Yoh_test = Y[index]

        s0 = np.zeros((1, n_s))
        c0 = np.zeros((1, n_s))
        y_pred = []
        for sample in Xoh_test:
            prediction = model.predict([np.expand_dims(sample, axis=0),np.expand_dims(sample, axis=0),np.expand_dims(sample, axis=0),np.expand_dims(sample, axis=0),np.expand_dims(sample, axis=0), s0, c0])
            y_pred.append(np.argmax(prediction[index].squeeze()))

        y_true = []
        for label in Yoh_test:
            y_true.append(np.argmax(label))

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        accs.append(round(acc,4))
        f1s.append(round(f1,4))

    return accs, f1s

def train_mtl(X, Y, Tx, Ty, n_a=32, n_s=64, out_dim = 2, dc_out_dim=15, wv_dim=300,epochs=10,drop=0.4,grl_lambda=0.4, dc_weight=0.4):

    #X, Y = upbalance_dataset(X,Y,dc_out_dim)

    X1 = X[0]
    X2 = X[1]
    X3 = X[2]
    X4 = X[3]
    X5 = X[4]
    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    Y4 = Y[3]
    Y5 = Y[4]

    min_num = min(len(X1), len(X2), len(X3), len(X4), len(X5))
    #(X1, Y1) = inplace_shuffle(X1, Y1)
    X1 = X1[:min_num]
    Y1 = Y1[:min_num]
    X2 = X2[:min_num]
    Y2 = Y2[:min_num]
    X3 = X3[:min_num]
    Y3 = Y3[:min_num]
    X4 = X4[:min_num]
    Y4 = Y4[:min_num]
    X5 = X5[:min_num]
    Y5 = Y5[:min_num]

    model = bilstm_attn_mtl(Tx, Ty, n_a, n_s, wv_dim, out_dim, dc_out_dim, drop=drop,grl_lambda=grl_lambda)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification_0': 'categorical_crossentropy','classification_1': 'categorical_crossentropy','classification_2': 'categorical_crossentropy','classification_3': 'categorical_crossentropy','domain_classification': 'categorical_crossentropy'},loss_weights={'classification_0': 1.0,'classification_1': 1.0,'classification_2': 1.0,'classification_3': 1.0,'domain_classification': dc_weight},optimizer=opt,metrics={'classification_0': 'accuracy','classification_1': 'accuracy','classification_2': 'accuracy','classification_3': 'accuracy','domain_classification': 'accuracy'})
 
    s0 = np.zeros((len(X1), n_s))
    c0 = np.zeros((len(X1), n_s))
    output1 = list(Y1.swapaxes(0,1))[0]
    output2 = list(Y2.swapaxes(0,1))[0]
    output3 = list(Y3.swapaxes(0,1))[0]
    output4 = list(Y4.swapaxes(0,1))[0]
    output5 = list(Y5.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    model.fit([X1, X2, X3, X4, X5, s0, c0], {'classification_0': output1,'classification_1': output2,'classification_2': output3,'classification_3': output4,'domain_classification': output5},batch_size=32,epochs=epochs,validation_split=0.15, shuffle=True, callbacks=callbacks,verbose=0)
    
    return model
