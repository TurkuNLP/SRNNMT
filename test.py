from keras.models import Sequential, Graph, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
from keras.layers.recurrent import GRU
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import numpy as np
import sys
import math
import json

import data_dense

# load model
with open("keras_model.json","r") as f:
    trained_model=model_from_json(f.read())
    trained_model.load_weights("keras_weights.h5")


#Read vocabularies
src_f_name="data/JRC-Acquis.en-fi.fi"
trg_f_name="data/JRC-Acquis.en-fi.en"
vs=data_dense.read_vocabularies(src_f_name,trg_f_name,False)
vs.trainable=False

minibatch_size=1
max_sent_len=200
vec_size=50
gru_width=50

ms=data_dense.Matrices(minibatch_size,max_sent_len)

#Input: sequences representing the source and target sentences
#Inputs
src_inp=Input(shape=(max_sent_len,), name="source_chars", dtype="int32")
trg_inp=Input(shape=(max_sent_len,), name="target_chars", dtype="int32")
#Embeddings
src_emb=Embedding(len(vs.source_chars), vec_size, input_length=max_sent_len, mask_zero=True, weights=trained_model.get_layer('src_embedding').get_weights())
trg_emb=Embedding(len(vs.target_chars), vec_size, input_length=max_sent_len, mask_zero=True, weights=trained_model.get_layer('trg_embedding').get_weights())
#Vectors
src_vec=src_emb(src_inp)
trg_vec=trg_emb(trg_inp)
#RNNs
src_gru=GRU(gru_width, weights=trained_model.get_layer('src_gru').get_weights())
trg_gru=GRU(gru_width, weights=trained_model.get_layer('trg_gru').get_weights())
src_gru_out=src_gru(src_vec)
trg_gru_out=trg_gru(trg_vec)
#Dense on top
src_dense=Dense(gru_width, weights=trained_model.get_layer('src_dense').get_weights())
trg_dense=Dense(gru_width, weights=trained_model.get_layer('trg_dense').get_weights())
src_dense_out=src_dense(src_gru_out)
trg_dense_out=trg_dense(trg_gru_out)
# ready...


model=Model(input=[src_inp,trg_inp], output=[src_dense_out,trg_dense_out])
model.compile(optimizer='adam',loss='mse')


test_size=1000

def iter_test_data(inf_i):
    for src_sent,trg_sent in inf_i.data[:test_size]:
        yield (src_sent,trg_sent),1.0

inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name)
batch_iter=data_dense.fill_batch(minibatch_size,max_sent_len,vs,inf_iter)

src_data=np.zeros((test_size,gru_width))
trg_data=np.zeros((test_size,gru_width))

for i,(mx,targets) in enumerate(data_dense.fill_batch(1,max_sent_len,vs,iter_test_data(inf_iter))):
    src,trg=model.predict(mx)
    src_data[i]=src[0]
    trg_data[i]=trg[0]
    if i>test_size:
        break

sims=trg_data.dot(src_data[0])
print(sims)
print(np.dot(src_data[0],trg_data[1]))
N=10
results=sorted(((sims[idx],idx,inf_iter.data[idx][1]) for idx in np.argpartition(sims,-N-1)[-N-1:]), reverse=True)
print(inf_iter.data[0][0])
for sim,idx,text in results:
    print(idx,sim,text)
sys.exit()

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


