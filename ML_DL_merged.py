'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from numpy import asarray
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU
from keras.layers import Concatenate, Input, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os, sys
import statistics
import random
from keras.optimizers import SGD, Adam
from imblearn.over_sampling import SMOTE
from numpy import zeros
import sklearn_crfsuite

from keras.layers.core import *
from keras.models import *
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk

filename = sys.argv[1]
file = open(filename, "r")
lines = file.read()
lines = lines.split("####SCORE###")
train_data = [lines[x] for x in range(len(lines)) if x%2 == 0]
y_labels = [lines[x] for x in range(len(lines)) if x%2 != 0]
y_labels = [float(x.strip()) for x in y_labels]
train_data = train_data[:-1]

final = []
for i in range(len(y_labels)):
    final.append((train_data[i], y_labels[i]))


random.shuffle(final)

train_data = []
y_labels = []
test_data = []
vali_data = []
y_test = []
vali_test = []

for i in range(0,len(final)-int(0.2*len(final))):
    train_data.append(final[i][0])
    y_labels.append(final[i][1])    


for i in range(len(final)-int(0.2*len(final)), len(final)-int(0.1*len(final))):
    vali_data.append(final[i][0])
    vali_test.append(final[i][1])


for i in range(len(final)-int(0.1*len(final)), len(final)):
    test_data.append(final[i][0])
    y_test.append(final[i][1])

    
for i in range(len(y_labels)):
    if(y_labels[i]<0):
        y_labels[i] = 0
    else:
        y_labels[i] = 1

        
for i in range(len(y_test)):
    if(y_test[i]<0):
        y_test[i] = 0
    else:
        y_test[i] = 1

for i in range(len(vali_test)):
    if(vali_test[i]<0):
        vali_test[i] = 0
    else:
        vali_test[i] = 1


feats_test = []
feats_train = []
feats_vali = []

feats_test = feature_extractor(test_data)
feats_train = feature_extractor(train_data)
feats_vali = feature_extractor(vali_data)


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 600
batch_size = 50

print('Loading data...')
# for i in range(len(train_data)):
#     if(train_data[i].split("#$#$#")[1].strip() == ''):
#         train_data[i] = train_data[i].split("#$#$#")[0].strip()
#     else:
#         train_data[i] = train_data[i].split("#$#$#")[1].strip()

# for i in range(len(test_data)):
#     if(test_data[i].split("#$#$#")[1].strip() == ''):
#         test_data[i] = test_data[i].split("#$#$#")[0].strip()
#     else:
#         test_data[i] = test_data[i].split("#$#$#")[1].strip()
# check = [i.split("#$#$#")[1].strip() for i in train_data]

lens = []
count = 0
for i in train_data:
    # print(i)
    count+=len(i)
    lens.append(len(i))

# print(count/len(train_data))
# print(statistics.mean(lens), statistics.stdev(lens))
# lens_set = sorted(list(set(lens)))
# counts = [lens.count(lens_set[0])]
# for i in range(1, len(lens_set)):
#     counts.append(lens.count(lens_set[i])+counts[-1])
# plt.plot(lens_set, counts, 'ro')
# plt.show()
# exit()

# exit()
embeddings_path = "glove.6B.100d-char.txt"
embeddings_dim = 100

text = ""
for i in train_data:
    text+=i

for i in test_data:
    text+=i

for i in vali_data:
    text+=i

# text = open('magic_cards.txt').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X_train = np.zeros((len(train_data), maxlen), dtype=np.int)
X_vali = np.zeros((len(vali_data), maxlen), dtype=np.int)
Y_train = np.zeros(len(train_data), dtype=int)
X_test = np.zeros((len(test_data), maxlen), dtype=np.int)
Y_test = np.zeros(len(test_data), dtype=int)
V_test = np.zeros(len(vali_data), dtype=int)

for i, sentence in enumerate(train_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_train[i, t] = char_indices[char]
    Y_train[i] = y_labels[i]

for i, sentence in enumerate(test_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_test[i, t] = char_indices[char]
    Y_test[i] = y_test[i]

for i, sentence in enumerate(vali_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_vali[i, t] = char_indices[char]
    V_test[i] = vali_test[i]

embedding_vectors = {}
with open(embeddings_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embedding_vectors[char] = vec

embedding_matrix = np.zeros((len(chars), embeddings_dim))
#embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
for char, i in char_indices.items():
    #print ("{}, {}".format(char, i))
    embedding_vector = embedding_vectors.get(char)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# =========================================
# =           Without Attention           =
# =========================================

e = Embedding(len(chars), embeddings_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)

sentence_input = Input(shape=(maxlen, ), dtype='int32')
embedded_sequences = e(sentence_input)

l_lstm = Bidirectional(GRU(20, return_sequences=False))(embedded_sequences)

second_input = Input(shape=(18, ))
merge_one = concatenate([l_lstm, second_input])

# lstm_out = Bidirectional(LSTM(20))
out1 = Dense(16, activation='relu')(merge_one)
# model.add(Dropout(0.5))
out2 = Dense(8, activation='relu')(out1)
out3 = Dropout(0.5)(out2)
out4 = Dense(4, activation='relu')(out3)
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='tanh'))
out5 = Dense(1, activation='sigmoid')(out4)

model = Model(inputs = [sentence_input, second_input], outputs = out5)
# ======  End of Without Attention  =======

# try using different optimizers and different optimizer configs
sgd = Adam(lr=0.001, decay=1e-4)
model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
model.fit([X_train, feats_train], Y_train, batch_size=batch_size, epochs=10,  validation_data=[[X_vali, feats_vali], V_test])

out = model.predict([X_test, feats_test])

count_1 = 0
count_0 = 0

pred_1 = []
for i in out:
    if(i[0] >= 0.50):
        pred_1.append([1])
    else:
        pred_1.append([0])

y_test_1 = []
for i in Y_test:
	y_test_1.append([int(i)])

true_1 = 0
true_0 = 0
false_1 = 0
false_0 = 0
for i in range(len(Y_test)):
	if((Y_test[i] == 1 and out[i][0] >= 0.50) ):
		true_1 += 1
	elif(Y_test[i] == 1 and out[i][0] < 0.50):
		false_1 += 1
	elif(Y_test[i] == 0 and out[i][0] < 0.50):
		true_0 += 1
	elif(Y_test[i] == 0 and out[i][0] >= 0.50):
		false_0 += 1
	

print("Class 1: True - "+str(true_1)+" False - "+str(false_1))
print("Class 0: True - "+str(true_0)+" False - "+str(false_0))
# 
# print(count_0, count_1)
print(out)

sorted_labels = [0, 1]
print(metrics.flat_classification_report(
	    y_test_1, pred_1, labels=sorted_labels, digits=3
	))

# print(count_false, count_true)

np.save("grounds", Y_test)
np.save("outs", out)
