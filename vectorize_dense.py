# python 3
# read finnish and english text file and turn these into dense vectors
from keras.models import Sequential, Graph, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
from keras.layers.recurrent import GRU
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import numpy as np
import sys
import math
import json
import gzip
import glob
import itertools
import pickle
import hashlib
import array

import data_dense
import re

from test import load_model

# TODO: expects 'fi' and 'en' prefixes in the input data_dir, could be changed to 'src' and 'trg'

            
def fill_batch(minibatch_size,max_sent_len,vs,data_iterator,ngrams):
    """ Iterates over the data_iterator and fills the index matrices with fresh data
        ms = matrices, vs = vocabularies
    """
    # custom fill_batch to return also sentences...

    ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
    batchsize,max_sentence_len=ms.source_ngrams[ngrams[0]].shape #just pick any one of these really
    row=0
    src_sents=[]
    trg_sents=[]
    for (sent_src,sent_target),target in data_iterator:
        src_sents.append(sent_src)
        trg_sents.append(sent_target)
        for N in ngrams:
            for j,ngram in enumerate(data_dense.ngram_iterator(sent_src,N,max_sent_len)):
                ms.source_ngrams[N][row,j]=vs.get_id(ngram,vs.source_ngrams[N])
            for j,ngram in enumerate(data_dense.ngram_iterator(sent_target,N,max_sent_len)):
                ms.target_ngrams[N][row,j]=vs.get_id(ngram,vs.target_ngrams[N])
        ms.src_len[row]=len(sent_src.strip().split())
        ms.trg_len[row]=len(sent_target.strip().split())
        ms.targets[row]=target
        row+=1
        if row==batchsize:
#            print(ms.matrix_dict, ms.targets)
            yield ms.matrix_dict, ms.targets, src_sents, trg_sents
            src_sents=[]
            trg_sents=[]
            row=0
            ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
    else:
        if row>0:
            yield ms.matrix_dict, ms.targets, src_sents, trg_sents
        

def iter_wrapper(fi,en,max_sent=10000):
    counter=0
    for fin_sent,eng_sent in itertools.zip_longest(fi,en,fillvalue="#None#"): # shorter padded with 'None'
        yield (fin_sent,eng_sent),1.0
        counter+=1
        if max_sent!=0 and counter==max_sent:
            break
        

def vectorize(args):

    # read and create files
    
    fi_inp=gzip.open(args.data_dir+"/fi_len{N}.txt.gz".format(N=args.length),"rt",encoding="utf-8")
    en_inp=gzip.open(args.data_dir+"/en_len{N}.txt.gz".format(N=args.length),"rt",encoding="utf-8")
    
    fi_outp=open(args.data_dir+"/fi_len{N}.npy".format(N=args.length),"wb")
    en_outp=open(args.data_dir+"/en_len{N}.npy".format(N=args.length),"wb")
    

    minibatch_size=100 
    ngrams=(4,) # TODO: read this from somewhere

    #Read vocabularies
    vs=data_dense.read_vocabularies(args.vocabulary,"xxx","xxx",False,ngrams) 
    vs.trainable=False
    
    # load model
    trained_model=load_model(args.model)
    output_size=trained_model.get_layer('source_dense').output_shape[1]
    max_sent_len=trained_model.get_layer('source_ngrams_{n}'.format(n=ngrams[0])).output_shape[1]
    print(output_size,max_sent_len,file=sys.stderr)
    
    # build matrices
    ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
    

    # get vectors
    # for loop over minibatches
    counter=0
    for i,(mx,targets,src_data,trg_data) in enumerate(fill_batch(minibatch_size,max_sent_len,vs,iter_wrapper(fi_inp,en_inp,max_sent=args.max_pairs),ngrams)):        
        src,trg=trained_model.predict(mx) # shape = (minibatch_size,gru_width)
        # loop over items in minibatch
        for j,(src_v,trg_v) in enumerate(zip(src,trg)):
            if j>=len(src_data): # empty padding of the minibatch
                break
            norm_src=src_v/np.linalg.norm(src_v)
            norm_trg=trg_v/np.linalg.norm(trg_v)
            if src_data[j]!="#None#":
                norm_src.astype(np.float32).tofile(fi_outp)
                    
            if trg_data[j]!="#None#":
                norm_trg.astype(np.float32).tofile(en_outp)
            
            counter+=1
            if counter%100000==0:
                print("Vectorized {c} sentence pairs".format(c=counter),file=sys.stderr,flush=True)
                
    fi_inp.close()
    en_inp.close()
    fi_outp.close()
    en_outp.close()


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--data_dir', type=str, help='Directory where text files are and where vectors should be written.')
    g.add_argument('-l', '--length', type=str, help='Sentence length, tells us which files to read')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give max pairs of sentences to read, zero for all, default={n}'.format(n=1000))
    
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None or args.length==None or args.data_dir==None:
        parser.print_help()
        sys.exit(1)

    number=str(args.max_pairs) if args.max_pairs!=0 else "all"
    print("Vectorizing",number,"sentences",file=sys.stderr)

    vectorize(args)

#    for s in iter_wrapper("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",max_sent=10000000):
#        pass




