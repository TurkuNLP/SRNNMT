from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten,ActivityRegularization
# from keras.layers.core import Masking
from keras.layers.recurrent import GRU
# from keras.optimizers import SGD
# from keras.datasets import reuters
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
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

model_name="gru_model"

minibatch_size=400
max_sent_len=200
vec_size=75
gru_width=75
ngrams=(4,)
ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
        
#Read vocabularies
src_f_name="data/all.train.fi.tokenized"
trg_f_name="data/all.train.en.tokenized"
vs=data_dense.read_vocabularies(model_name+"-vocab.pickle",src_f_name,trg_f_name,False,ngrams)
vs.trainable=False

#Inputs: list of one Input per N-gram size
src_inp=[Input(shape=(max_sent_len,), name="source_ngrams_{}".format(N), dtype="int32") for N in ngrams]
trg_inp=[Input(shape=(max_sent_len,), name="target_ngrams_{}".format(N), dtype="int32") for N in ngrams]
# sent len
src_len_inp=Input(shape=(1,), name="src_len", dtype="int32")
trg_len_inp=Input(shape=(1,), name="trg_len", dtype="int32")

#Embeddings: list of one Embedding per input
src_emb=[Embedding(len(vs.source_ngrams[N]), vec_size, input_length=max_sent_len, mask_zero=True, name="source_embedding_{}".format(N)) for N in ngrams]
trg_emb=[Embedding(len(vs.target_ngrams[N]), vec_size, input_length=max_sent_len, mask_zero=True, name="target_embedding_{}".format(N)) for N in ngrams]
# sent len
flattener1=Flatten()
flattener2=Flatten()
src_len_emb=Embedding(31,vec_size, input_length=1, name="src_len_emb")
#src_flattener=Flatten()
src_len_vec=flattener1(src_len_emb(src_len_inp))
trg_len_emb=Embedding(31,vec_size, input_length=1, name="trg_len_emb")
#trg_flattener=Flatten()
trg_len_vec=flattener2(trg_len_emb(trg_len_inp))

#Vectors: list of one embedded vector per input-embedding pair
src_vec=[src_emb_n(src_inp_n) for src_inp_n,src_emb_n in zip(src_inp,src_emb)]
trg_vec=[trg_emb_n(trg_inp_n) for trg_inp_n,trg_emb_n in zip(trg_inp,trg_emb)]

#RNNs: list of one GRU per ngram size
# forward
src_gru=[GRU(gru_width,consume_less="gpu",dropout_W=0.3,activation="relu",return_sequences=False,name="source_GRU_{}".format(N)) for N in ngrams]
trg_gru=[GRU(gru_width,consume_less="gpu",dropout_W=0.3,activation="relu",return_sequences=False,name="target_GRU_{}".format(N)) for N in ngrams]
src_gru_out=[src_gru_n(src_vec_n) for src_vec_n,src_gru_n in zip(src_vec,src_gru)]
trg_gru_out=[trg_gru_n(trg_vec_n) for trg_vec_n,trg_gru_n in zip(trg_vec,trg_gru)]

#Catenate the GRUs
src_gru_all=merge(src_gru_out+[src_len_vec],mode='concat',concat_axis=1,name="src_gru_concat")
trg_gru_all=merge(trg_gru_out+[trg_len_vec],mode='concat',concat_axis=1,name="trg_gru_concat")

src_dense_lin_out=Dense(gru_width,name="source_dense")(src_gru_all)
trg_dense_lin_out=Dense(gru_width,name="target_dense")(trg_gru_all)


#..regularize
#src_dense_reg=ActivityRegularization(l2=1.0,name="source_dense")
#trg_dense_reg=ActivityRegularization(l2=1.0,name="target_dense")
#src_dense_reg_out=src_dense_reg(src_dense_out)
#trg_dense_reg_out=trg_dense_reg(trg_dense_out)

#...and cosine between the source and target side
merged_out=merge([src_dense_lin_out,trg_dense_lin_out],mode='cos',dot_axes=1)
flatten=Flatten()
merged_out_flat=flatten(merged_out)

model=Model(input=src_inp+trg_inp+[src_len_inp,trg_len_inp], output=merged_out_flat)
model.compile(optimizer='adam',loss='mse')
print(model.summary())

inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name)
batch_iter=data_dense.fill_batch(minibatch_size,max_sent_len,vs,inf_iter,ngrams)

#dev iter
dev_batch_iter=data_dense.fill_batch(minibatch_size,max_sent_len,vs,data_dense.InfiniteDataIterator("data/all.dev.new.fi.tokenized","data/all.dev.new.en.tokenized"),ngrams)

# import pdb
# pdb.set_trace()

# save model json
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)

# callback to save weights after each epoch
save_cb=ModelCheckpoint(filepath=model_name+".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

samples_per_epoch=math.ceil((2*len(inf_iter.data))/minibatch_size/20)*minibatch_size #2* because we also have the negative examples
model.fit_generator(batch_iter,samples_per_epoch,60,callbacks=[save_cb],validation_data=dev_batch_iter,nb_val_samples=1000)

#counter=1
#while True:
#    matrix_dict,target=batch_iter.__next__()
#    print("BATCH", counter, "LOSS",model.train_on_batch(matrix_dict,target),file=sys.stderr,flush=True)
#    counter+=1
