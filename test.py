import logging
logging.getLogger('tensorflow').disabled = True # this removes the annoying 'Level 1:tensorflow:Registering' prints

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
from keras.layers.recurrent import GRU
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import numpy as np
import sys
import math
import json
import os
from keras.preprocessing.sequence import pad_sequences

import data_dense

import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
### ---end of weird stuff

# load model
def load_model(mname):
    with open(mname+"-source_encoder.json","r") as f:
        source_model=model_from_json(f.read())
        source_model.load_weights(mname+"-source_encoder.h5")
        print(source_model.summary())
        print(source_model.outputs)
    with open(mname+"-target_encoder.json","r") as f:
        target_model=model_from_json(f.read())
        target_model.load_weights(mname+"-target_encoder.h5")
        print(target_model.summary())
        print(target_model.outputs)
        
    return source_model, target_model


def data_vectorizer(batch, max_seq_len, vs, src_data,trg_data):
    src_sentences=[]
    trg_sentences=[]
    for src_sent,trg_sent in zip(src_data,trg_data):
        s,t=vs.sentence_vectorizer(src_sent,trg_sent)
        src_sentences.append(s)
        trg_sentences.append(t)
        if len(src_sentences)==batch:
            src_data=pad_sequences(np.array(src_sentences), maxlen=max_seq_len, padding='post', truncating='post')
            trg_data=pad_sequences(np.array(trg_sentences), maxlen=max_seq_len, padding='post', truncating='post')
            yield src_data,trg_data
            src_sentences=[]
            trg_sentences=[]


def vectorize(src_data,trg_data,mname):

    minibatch_size=10 

    #Read vocabularies
    vs=data_dense.WhitespaceSeparatedVocab()
    vs.load_vocabularies(mname+"-vocab.json") 
    vs.trainable=False
    
    # load model
    source_model, target_model=load_model(mname)
    #output_size=source_model.get_layer('source_dense').output_shape[1]
    #max_sent_len=trained_model.get_layer('source_ngrams_{n}'.format(n=ngrams[0])).output_shape[1]
    max_seq_len=50 # TODO
    
    
    src_vectors=np.zeros((len(src_data),1024))
    trg_vectors=np.zeros((len(src_data),1024))

    # get vectors
    # for loop over minibatches
    counter=0    
    for i,(src,trg) in enumerate(data_vectorizer(minibatch_size,max_seq_len,vs,src_data,trg_data)):
        src=source_model.predict(src)
        trg=source_model.predict(trg)
        # loop over items in minibatch
        for j,(src_v,trg_v) in enumerate(zip(src,trg)):
            src_vectors[counter]=src_v/np.linalg.norm(src_v)
            trg_vectors[counter]=trg_v/np.linalg.norm(trg_v)
            counter+=1
            if counter==len(src_data):
                break
        if counter==len(src_data):
            break
    return src_vectors,trg_vectors
    
    
def rank(src_vectors,trg_vectors,src_data,trg_data,verbose=True):

    ranks=[]
    all_similarities=[] # list of sorted lists

    # run dot product
    for i in range(len(src_vectors)):
        sims=src_vectors.dot(trg_vectors[i])
        all_similarities.append(sims)  
        N=10
        results=sorted(((sims[idx],idx,trg_data[idx]) for idx,s in enumerate(sims)), reverse=True)
#        if results[0][0]<0.6:
#            continue
        result_idx=[idx for (sim,idx,txt) in results]
        ranks.append(result_idx.index(i)+1)
        if verbose:
            print("source:",i,src_data[i],np.dot(src_vectors[i],trg_vectors[i]))
            print("reference:",trg_data[i])
            print("rank:",result_idx.index(i)+1)
            for s,idx,txt in results[:10]:
                print(idx,s,txt)
            print("****")

    print("Keras:")
    print("Avg:",sum(ranks)/len(ranks))
    print("#num:",len(ranks))
    
    return all_similarities
    
    
def test(src_fname,trg_fname,mname,args):

    # read sentences
    src_data=[]
    trg_data=[]
    for i,(src_line,trg_line) in enumerate(zip(open(src_fname),open(trg_fname))):
        src_data.append(src_line.strip())
        trg_data.append(trg_line.strip())
        if args.max_pairs!=0 and i>=args.max_pairs-1:
            break
        
    src_vectors,trg_vectors=vectorize(src_data,trg_data,mname)
    if args.monolingual_source:
        similarities=rank(src_vectors,src_vectors,src_data,src_data,verbose=args.verbose)
    elif args.monolingual_target:
        similarities=rank(trg_vectors,trg_vectors,trg_data,trg_data,verbose=args.verbose)
    else: # normal multilingual evaluation
        similarities=rank(src_vectors,trg_vectors,src_data,trg_data,verbose=args.verbose)

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    g.add_argument('--verbose', action='store_true', default=False, help='Give vocabulary file')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, default={n}'.format(n=1000))
    g.add_argument('--monolingual_source', action='store_true', default=False, help='Run monolingual evaluation (src against src similarities)')
    g.add_argument('--monolingual_target', action='store_true', default=False, help='Run monolingual evaluation (trg against trg similarities)')
    
    args = parser.parse_args()

    if args.model==None:
        parser.print_help()
        sys.exit(1)

    src_file="data/devel_data/newstest2015.fi.subwords"
    trg_file="data/devel_data/newstest2015.en.subwords"

#    src_file="data/europarl-v7.fi-en.fi"
#    trg_file="data/europarl-v7.fi-en.en"


    test(src_file,trg_file,args.model,args)
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


