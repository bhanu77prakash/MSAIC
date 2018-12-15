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
import sys
from data_preprocess import preprocess
import os

# ====================================
# =           Loading Data           =
# ====================================

if len(sys.argv)!=2:
    print("Usage: python train.py [mode train_data.tsv or data.tsv]")
    exit()

tr_dat = sys.argv[1]
conditions = os.path.isfile("./data/preprocessed_train_pasges("+tr_dat+").npy") and os.path.isfile("./data/preprocessed_train_queries("+tr_dat+").npy") and os.path.isfile("./data/preprocessed_test_queries("+tr_dat+").npy") and os.path.isfile("./data/preprocessed_test_pasges("+tr_dat+").npy") and os.path.isfile("./data/embed_word_matrix("+tr_dat+").npy") and os.path.isfile("./data/labels("+tr_dat+").npy") and os.path.isfile("./data/embed_char_matrix("+tr_dat+").npy")
if(conditions == False):
    preprocess(tr_dat)

queries = np.loadtxt("./data/preprocessed_train_queries("+tr_dat+").npy")
pasges = np.loadtxt("./data/preprocessed_train_pasges("+tr_dat+").npy")
test_queries = np.loadtxt("./data/preprocessed_test_queries("+tr_dat+").npy")
test_pasges = np.loadtxt("./data/preprocessed_test_pasges("+tr_dat+").npy")
labels = np.loadtxt("./data/labels("+tr_dat+").npy")
embed_word_matrix = np.loadtxt("./data/embed_word_matrix("+tr_dat+").npy")
embed_char_matrix = np.loadtxt("./data/embed_char_matrix("+tr_dat+").npy")
df = pd.read_csv("./data/eval1_unlabelled.tsv", sep="\t", header=None)   # read dummy .tsv file into memory
test_data = df.values

# ======  End of Loading Data  =======


# ========================================
# =           Hyper Parameters           =
# ========================================

embed_char_dim = 100
embed_word_dim = embed_char_dim
query_maxlen = 12
passage_maxlen = 50
max_features = 20000
batch_size = 250

# ======  End of Hyper Parameters  =======


# ===========================================
# =           Attention mechanism 1          =
# ===========================================


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# ======  End of Attention mechanism 1 =======

# =============================================
# =           Attention mechanism 2           =
# =============================================
repeator = RepeatVector(passage_maxlen)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(activation = "softmax", name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
    s_prev = repeator(s_prev) 
    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a,s_prev]) 
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = densor1(concat) 
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e) 
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies) 
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas,a]) 
    ### END CODE HERE ###
    
    return Reshape((40,))(context)

# ======  End of Attention mechanism 2  =======


# =============================
# =           Model           =
# =============================
vocab_size = len(embed_word_matrix)

e_pasge = Embedding(vocab_size, embed_word_dim, weights=[embed_word_matrix], input_length=passage_maxlen, trainable=False)
e_query = Embedding(vocab_size, embed_word_dim, weights=[embed_word_matrix], input_length=query_maxlen, trainable=False)

pasge_input = Input(shape=(passage_maxlen, ))
pasge_embed_seq = e_pasge(pasge_input)
query_input = Input(shape=(query_maxlen, ))
query_embed_seq = e_query(query_input)

lstm_query = Bidirectional(GRU(20, return_sequences=False))(query_embed_seq)
lstm_pasge = Bidirectional(GRU(20, return_sequences=True))(pasge_embed_seq)

attend_pasge = one_step_attention(lstm_pasge, lstm_query)
merge_one = concatenate([lstm_query, attend_pasge])
out1 = Dense(64, activation='tanh')(merge_one)
out2 = Dropout(0.2)(out1)
out3 = Dense(16, activation='tanh')(out2)
out4 = Dropout(0.2)(out3)
out5 = Dense(4, activation='tanh')(out4)
out6 = Dense(1, activation='sigmoid')(out5)
model = Model(inputs = [pasge_input, query_input], outputs = out6)

# ======  End of Model  =======

# ======================================
# =           Training Model           =
# ======================================

sgd = Adam(lr=0.001, decay=1e-4)
model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
model.fit([pasges, queries], labels, batch_size=batch_size, epochs=5)
model.save('model_5_epochs('+tr_dat+').h5')

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
