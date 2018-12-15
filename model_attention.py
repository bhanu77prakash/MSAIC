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
from keras.layers import Permute, Dot, Input, Multiply
from keras.layers import RepeatVector, Activation, Lambda

# ====================================
# =           Loading Data           =
# ====================================

tr_dat = "train_data.tsv"
df = pd.read_csv(tr_dat, sep="\t", header=None)   # read dummy .tsv file into memory

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

query_maxlen = 20
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


np.savetxt("preprocessed_train_pasges("+tr_dat+").npy", pasges)
np.savetxt("preprocessed_train_queries("+tr_dat+").npy", queries)
np.savetxt("preprocessed_test_queries("+tr_dat+").npy", test_queries)
np.savetxt("preprocessed_test_pasges("+tr_dat+").npy", test_pasges)
np.savetxt("embed_word_matrix("+tr_dat+").npy", embed_word_matrix)
labels = train_data[:, 3]
np.savetxt("labels("+tr_dat+").npy", labels)

# ======  End of Tokenizing the training data  =======


