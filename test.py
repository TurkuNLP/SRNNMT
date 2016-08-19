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
#from train import max_sent_len, vec_size, gru_width, ngrams

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


minibatch_size=100
#max_sent_len=200
#vec_size=100
#gru_width=100
#ngrams=(3,4,5)
max_sent_len=30
vec_size=100
gru_width=100
ngrams=(3,4,5)

#Read vocabularies
src_f_name="data/Europarl.en-fi.fi"
trg_f_name="data/Europarl.en-fi.en"
vs=data_dense.read_vocabularies("data/JRC-Acquis.en-fi.fi","data/JRC-Acquis.en-fi.en",False,ngrams) 
vs.trainable=False

ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)


test_size=1000

data=[]
def iter_wrapper(src_fname,trg_fname):
    for src_sent,trg_sent in data_dense.iter_data(src_fname,trg_fname):
        data.append((src_sent,trg_sent))
        yield (src_sent,trg_sent),1.0



src_data=np.zeros((test_size,gru_width))
trg_data=np.zeros((test_size,gru_width))


counter=0
# for loop over minibatches
for i,(mx,targets) in enumerate(data_dense.fill_batch(minibatch_size,max_sent_len,vs,iter_wrapper(src_f_name,trg_f_name),ngrams)):
    src,trg=trained_model.predict(mx) # shape = (minibatch_size,gru_width)
    # loop over items in minibatch
    for j,(src_v,trg_v) in enumerate(zip(src,trg)):
        src_data[counter]=src_v/np.linalg.norm(src_v)
        trg_data[counter]=trg_v/np.linalg.norm(trg_v)
        counter+=1
        if counter==test_size:
            break
    if counter==test_size:
        break

ranks=[]
verbose=True

# run dot product
for i in range(test_size):
    sims=trg_data.dot(src_data[i])  
    N=10
    results=sorted(((sims[idx],idx,data[idx][1]) for idx in np.argpartition(sims,-N-1)), reverse=True)#[-N-1:]), reverse=True)
#    if results[0][0]<0.6:
#        continue
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
print("#num:",len(ranks))



#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


