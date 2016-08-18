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


test_size=100

data=[]
def iter_wrapper(src_fname,trg_fname):
    for src_sent,trg_sent in data_dense.iter_data(src_fname,trg_fname):
        data.append((src_sent,trg_sent))
        yield (src_sent,trg_sent),1.0


src_data=np.zeros((test_size,gru_width*len(ngrams)))
trg_data=np.zeros((test_size,gru_width*len(ngrams)))

# fill in src and trg matrices
for i,(mx,targets) in enumerate(data_dense.fill_batch(1,max_sent_len,vs,iter_wrapper(src_f_name,trg_f_name),ngrams)):
    src,trg=trained_model.predict(mx) # shape = (1,150)
    src_data[i]=src[0]
    trg_data[i]=trg[0]
    if i==test_size-1:
        break

ranks=[]
verbose=False

# run dot product
for i in range(test_size):
    sims=trg_data.dot(src_data[i])  
    N=10
    results=sorted(((sims[idx],idx,data[idx][1]) for idx in np.argpartition(sims,-N-1)), reverse=True)#[-N-1:]), reverse=True)
    result_idx=[idx for (sim,idx,txt) in results]
    ranks.append(result_idx.index(i)+1)
    if verbose:
        print("source:",i,data[i][0].strip(),np.dot(src_data[i],trg_data[i]))
        print("reference:",data[i][1].strip())
        print("rank:",result_idx.index(i)+1)
        for s,idx,txt in results[:10]:
            print(idx,s,txt)
        print("****")

print("Avg:",sum(ranks)/len(ranks))



#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


