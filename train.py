from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
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
# import sys
# from svm_pronouns import iter_data
# import json
# import copy
# from data_dense import *
# from sklearn.metrics import recall_score

import data_dense

class CustomCallback(Callback):

    def __init__(self, dev_data,dev_labels,index2label,model_name):
        pass
        # self.model_name = model_name
        # self.dev_data=dev_data
        # self.dev_labels=dev_labels
        # self.index2label=index2label
        # self.best_mr = 0.0
        # self.dev_labels_text=[]
        # for l in self.dev_labels:
        #     self.dev_labels_text.append(index2label[np.argmax(l)])

    def on_epoch_end(self, epoch, logs={}):
        pass
        # print logs

        # corr=0
        # tot=0
        # preds = self.model.predict(self.dev_data, verbose=1)
        # preds_text=[]
        # for l in preds:
        #     preds_text.append(self.index2label[np.argmax(l)])

        # print "Micro f-score:", f1_score(self.dev_labels_text,preds_text,average=u"micro")
        # print "Macro f-score:", f1_score(self.dev_labels_text,preds_text,average=u"macro")
        # print "Macro recall:", recall_score(self.dev_labels_text,preds_text,average=u"macro")

        # if self.best_mr < recall_score(self.dev_labels_text,preds_text,average=u"macro"):
        #     self.best_mr = recall_score(self.dev_labels_text,preds_text,average=u"macro")
        #     model.save_weights('./models_gru/' + self.model_name + '_' + str(epoch) + '_MR_' + str(self.best_mr) + '.hdf5')
        #     print 'Saved Weights!'


        # print classification_report(self.dev_labels_text, preds_text)
        # for i in xrange(len(self.dev_labels)):

        # #    next_index = sample(preds[i])
        #     next_index = np.argmax(preds[i])
        #     # print preds[i],next_index,index2label[next_index]

        #     l = self.index2label[next_index]

        #     # print "correct:", index2label[np.argmax(dev_labels[i])], "predicted:",l
        #     if self.index2label[np.argmax(self.dev_labels[i])]==l:
        #         corr+=1
        #     tot+=1
        # print corr,"/",tot




#Read vocabularies
src_f_name="data/JRC-Acquis.en-fi.fi"
trg_f_name="data/JRC-Acquis.en-fi.en"
vs=data_dense.read_vocabularies(src_f_name,trg_f_name,False)
vs.trainable=False

minibatch_size=5000
max_sent_len=500
vec_size=100
gru_width=100

ms=data_dense.Matrices(minibatch_size,max_sent_len)

#Input: sequences representing the source and target sentences
#Inputs
src_inp=Input(shape=(max_sent_len,), name="source_chars", dtype="int32")
trg_inp=Input(shape=(max_sent_len,), name="target_chars", dtype="int32")
#Embeddings
src_emb=Embedding(len(vs.source_chars), vec_size, input_length=max_sent_len, mask_zero=True)
trg_emb=Embedding(len(vs.target_chars), vec_size, input_length=max_sent_len, mask_zero=True)
#Vectors
src_vec=src_emb(src_inp)
trg_vec=trg_emb(trg_inp)
#RNNs
src_gru=GRU(gru_width)
trg_gru=GRU(gru_width)
src_gru_out=src_gru(src_vec)
trg_gru_out=trg_gru(trg_vec)
#Output as a single vector, internal states of GRUs who have now read the data
merged_out=merge([src_gru_out,trg_gru_out],mode='cos',dot_axes=1)

model=Model(input=[src_inp,trg_inp], output=merged_out)
model.compile(optimizer='adam',loss='mse')

inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name,max_iterations=None)
batch_iter=data_dense.fill_batch(ms,vs,inf_iter)
model.fit_generator(batch_iter,2*len(inf_iter.data),10) #2* because we also have the negative examples

