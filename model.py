import pandas as pd 
import numpy as np 
from numpy import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU
from keras.optimizers import SGD, Adam
from keras.layers import Concatenate, Input, concatenate
from keras.layers.core import *
from keras.models import *

# ====================================
# =           Loading Data           =
# ====================================

df = pd.read_csv("data.tsv", sep="\t", header=None)   # read dummy .tsv file into memory

train_data = df.values  # access the numpy array containing values
print("Loaded training data")

np.random.shuffle(train_data)

df = pd.read_csv("eval1_unlabelled.tsv", sep="\t", header=None)   # read dummy .tsv file into memory

test_data = df.values  # access the numpy array containing values
print("Loaded testing data")

# ======  End of Loading Data  =======


embed_char_path = "glove.6B.100d-char.txt"
embed_char_dim = 100

embed_word_path = "glove.6B.100d.txt"
embed_word_dim = embed_char_dim

# ========================================
# =           Hyper Parameters           =
# ========================================

query_maxlen = 50
passage_maxlen = 50
max_features = 20000
batch_size = 50

# ======  End of Hyper Parameters  =======

# ====================================================
# =           Loading character embeddings           =
# ====================================================

total_chars = ''
for i in train_data:
	total_chars += i[1]
	total_chars += ' '
	total_chars += i[2]
	total_chars += ' '

chars = sorted(list(set(total_chars)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

embed_char_vectors = {}
with open(embed_char_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embed_char_vectors[char] = vec

embed_char_matrix = np.zeros((len(chars), embed_char_dim))
for char, i in char_indices.items():
    embedding_vector = embed_char_vectors.get(char)
    if embedding_vector is not None:
        embed_char_matrix[i] = embedding_vector
print("Loaded character embeddings")

# ======  End of Loading character embeddings  =======

# ===============================================
# =           Loading Word embeddings           =
# ===============================================

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data[:, 1])
tokenizer.fit_on_texts(train_data[:, 2])
tokenizer.fit_on_texts(test_data[:, 1])
tokenizer.fit_on_texts(test_data[:, 2])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
vocab_size = len(tokenizer.word_index) + 1

embed_word_index = dict()

f = open(embed_word_path)
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embed_word_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embed_word_index))
embed_word_matrix = zeros((vocab_size, embed_word_dim))
for word, i in tokenizer.word_index.items():
	embed_word_vector = embed_word_index.get(word)
	if embed_word_vector is not None:
		embed_word_matrix[i] = embed_word_vector
print("Loaded word embeddings")

# ======  End of Loading Word embeddings  =======

# ====================================================
# =           Tokenizing the training data           =
# ====================================================

queries = train_data[:, 1]
pasges = train_data[:, 2]
queries = tokenizer.texts_to_sequences(queries)
pasges = tokenizer.texts_to_sequences(pasges)
test_queries = test_data[:, 1]
test_pasges = test_data[:, 2]
test_queries = tokenizer.texts_to_sequences(test_queries)
test_pasges = tokenizer.texts_to_sequences(test_pasges)
queries = sequence.pad_sequences(queries, maxlen=query_maxlen)
pasges = sequence.pad_sequences(pasges, maxlen=passage_maxlen)

test_queries = sequence.pad_sequences(test_queries, maxlen=query_maxlen)
test_pasges = sequence.pad_sequences(test_pasges, maxlen=passage_maxlen)

np.savetxt("preprocessed_train_pasges.npy", pasges)
np.savetxt("preprocessed_train_queries.npy", queries)
np.savetxt("preprocessed_test_queries.npy", test_queries)
np.savetxt("preprocessed_test_pasges.npy", test_pasges)

labels = train_data[:, 3]

# ======  End of Tokenizing the training data  =======

# =============================
# =           Model           =
# =============================

e_pasge = Embedding(vocab_size, embed_word_dim, weights=[embed_word_matrix], input_length=passage_maxlen, trainable=False)
e_query = Embedding(vocab_size, embed_word_dim, weights=[embed_word_matrix], input_length=query_maxlen, trainable=False)

pasge_input = Input(shape=(passage_maxlen, ))
pasge_embed_seq = e_pasge(pasge_input)
query_input = Input(shape=(query_maxlen, ))
query_embed_seq = e_query(query_input)

lstm_query = Bidirectional(GRU(20, return_sequences=False))(query_embed_seq)
lstm_pasge = Bidirectional(GRU(20, return_sequences=False))(pasge_embed_seq)

merge_one = concatenate([lstm_query, lstm_pasge])
out1 = Dense(16, activation='relu')(merge_one)
out2 = Dense(8, activation='relu')(out1)
out3 = Dropout(0.5)(out2)
out4 = Dense(4, activation='relu')(out3)
out5 = Dense(1, activation='sigmoid')(out4)
model = Model(inputs = [pasge_input, query_input], outputs = out5)

# ======  End of Model  =======

# ======================================
# =           Training Model           =
# ======================================

sgd = Adam(lr=0.001, decay=1e-4)
model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
model.fit([pasges, queries], labels, batch_size=batch_size, epochs=5)
model.save('model_2_epochs.h5')

# ======  End of Training Model  =======

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# =========================================
# =           Testing the model           =
# =========================================

outs = model.predict([test_pasges, test_queries])
outputs = np.zeros((len(test_queries)/10, 11))
temp = []
for i in range(len(outs)):
	temp.append(outs[i])
	if (i+1)%10 == 0:
		values = softmax(temp)
		values = values.reshape(10,)
		outputs[i/10][0] = test_data[i][0]
		outputs[i/10][1:] = values
		temp = []

np.savetxt("answer.tsv", outputs, delimiter="\t", fmt='%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f')

# ======  End of Testing the model  =======