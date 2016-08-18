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
    trained_model.layers.pop() # remove cosine and flatten layers
    trained_model.layers.pop()
    trained_model.outputs = [trained_model.get_layer('source_dense').output,trained_model.get_layer('target_dense').output] # define new outputs
    trained_model.layers[-1].outbound_nodes = [] # not sure if we need these...
    trained_model.layers[-2].outbound_nodes = []
    print(trained_model.summary())
    print(trained_model.outputs)


minibatch_size=1
max_sent_len=200
vec_size=50
gru_width=50
ngrams=(3,4,5)

#Read vocabularies
src_f_name="data/JRC-Acquis.en-fi.fi"
trg_f_name="data/JRC-Acquis.en-fi.en"
vs=data_dense.read_vocabularies(src_f_name,trg_f_name,False,ngrams)
vs.trainable=False

ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)


test_size=1000

def iter_test_data(inf_i):
    for src_sent,trg_sent in inf_i.data[:test_size]:
        yield (src_sent,trg_sent),1.0

inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name)
batch_iter=data_dense.fill_batch(minibatch_size,max_sent_len,vs,inf_iter,ngrams)

src_data=np.zeros((test_size,gru_width*len(ngrams)))
trg_data=np.zeros((test_size,gru_width*len(ngrams)))

for i,(mx,targets) in enumerate(data_dense.fill_batch(1,max_sent_len,vs,iter_test_data(inf_iter),ngrams)):
    src,trg=trained_model.predict(mx)
    src_data[i]=src[0]
    trg_data[i]=trg[0]
    if i>test_size:
        break

sims=trg_data.dot(src_data[0])
print(np.dot(src_data[0],trg_data[0]))
N=10
results=sorted(((sims[idx],idx,inf_iter.data[idx][1]) for idx in np.argpartition(sims,-N-1)[-N-1:]), reverse=True)
print(inf_iter.data[0][0])
for sim,idx,text in results:
    print(idx,sim,text)
sys.exit()

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


