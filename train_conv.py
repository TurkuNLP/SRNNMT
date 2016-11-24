from keras.models import Sequential, Graph, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten,ActivityRegularization,Convolution1D,MaxPooling1D
# from keras.layers.core import Masking
from keras.layers.recurrent import GRU
# from keras.optimizers import SGD
# from keras.datasets import reuters
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.engine.topology import Layer

# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import f1_score, classification_report
# import codecs
# import numpy as np
# import gzip
import sys
import math
# from svm_pronouns import iter_data
import json
# import copy
# from data_dense import *
# from sklearn.metrics import recall_score

import data_dense

class CustomCallback(Callback):

    def __init__(self, dev_data,dev_labels,index2label,model_name):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass
        


# model name
if len(sys.argv)<2:
    print("Error: Give model name.")
    sys.exit(1)
    
model_name=sys.argv[1]

minibatch_size=400
max_sent_len=200
vec_size=150
gru_width=150
ngrams=(4,)
ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
        
#Read vocabularies
#src_f_name="data/all.train.fi.tokenized"
#trg_f_name="data/all.train.en.tokenized"
src_f_name="bible_data/bible.en-fi.fi.tok.train"
trg_f_name="bible_data/bible.en-fi.en.tok.train"
vs=data_dense.read_vocabularies(model_name+"-vocab.pickle",src_f_name,trg_f_name,False,ngrams)
vs.trainable=False

#Inputs: list of one Input per N-gram size
src_inp=Input(shape=(max_sent_len,), name="source_ngrams_{N}".format(N=ngrams[0]), dtype="int32")
trg_inp=Input(shape=(max_sent_len,), name="target_ngrams_{N}".format(N=ngrams[0]), dtype="int32")

#Embeddings: list of one Embedding per input
src_emb=Embedding(len(vs.source_ngrams[ngrams[0]]), vec_size, input_length=max_sent_len, name="source_embedding_{N}".format(N=ngrams[0]))(src_inp)
trg_emb=Embedding(len(vs.target_ngrams[ngrams[0]]), vec_size, input_length=max_sent_len, name="target_embedding_{N}".format(N=ngrams[0]))(trg_inp)


# Conv
src_conv_out=Convolution1D(vec_size, 5, border_mode="same", activation="relu")(src_emb) # output shape=(number of timesteps, vec_size)
trg_conv_out=Convolution1D(vec_size, 5, border_mode="same", activation="relu")(trg_emb)

src_maxpool_out=MaxPooling1D(pool_length=max_sent_len)(src_conv_out)
trg_maxpool_out=MaxPooling1D(pool_length=max_sent_len)(trg_conv_out)


src_flat_out=Flatten()(src_maxpool_out)
trg_flat_out=Flatten()(trg_maxpool_out)


# yet one dense
src_dense_out=Dense(gru_width,name="source_dense")(src_flat_out)
trg_dense_out=Dense(gru_width,name="target_dense")(trg_flat_out)


#...and cosine between the source and target side
merged_out=merge([src_dense_out,trg_dense_out],mode='cos',dot_axes=1)
flatten=Flatten()
merged_out_flat=flatten(merged_out)

model=Model(input=[src_inp,trg_inp], output=merged_out_flat)

model.compile(optimizer='adam',loss='mse')
print(model.summary())

inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name)
batch_iter=data_dense.fill_batch(minibatch_size,max_sent_len,vs,inf_iter,ngrams)

#dev iter
dev_batch_iter=data_dense.fill_batch(minibatch_size,max_sent_len,vs,data_dense.InfiniteDataIterator("bible_data/bible.en-fi.fi.tok.devel","bible_data/bible.en-fi.en.tok.devel"),ngrams)

# import pdb
# pdb.set_trace()

# save model json
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)

# callback to save weights after each epoch
save_cb=ModelCheckpoint(filepath=model_name+".{epoch:02d}.h5", monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

samples_per_epoch=math.ceil((2*len(inf_iter.data))/minibatch_size/20)*minibatch_size #2* because we also have the negative examples
model.fit_generator(batch_iter,samples_per_epoch,60*2,callbacks=[save_cb],validation_data=dev_batch_iter,nb_val_samples=1062)

#counter=1
#while True:
#    matrix_dict,target=batch_iter.__next__()
#    print("BATCH", counter, "LOSS",model.train_on_batch(matrix_dict,target),file=sys.stderr,flush=True)
#    counter+=1
