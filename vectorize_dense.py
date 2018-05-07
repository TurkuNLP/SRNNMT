# python 3
# read finnish and english text file and turn these into dense vectors

import logging
logging.getLogger('tensorflow').disabled = True # this removes the annoying 'Level 1:tensorflow:Registering' prints

from keras.models import Sequential, Model, model_from_json
#from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
#from keras.layers.recurrent import GRU
#from keras.callbacks import Callback,ModelCheckpoint
#from keras.layers.embeddings import Embedding
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

from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
### ---end of weird stuff

# TODO: expects 'fi' and 'en' prefixes in the input data_dir, could be changed to 'src' and 'trg'


def load_model(args):
    with open(args.model+".json","r") as f:
        encoder_model=model_from_json(f.read())
        encoder_model.load_weights(args.model+".h5")
        print(encoder_model.summary())
        print(encoder_model.outputs)
        
    return encoder_model



def data_vectorizer(batch, max_seq_len, vs, data_file, args):

    if args.subword_model is not None: # init sp model
        print("loading subword model from", args.subword_model+".model")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(args.subword_model+".model")
    else:
        assert False, "Remember to give subword model!" # TODO

    sentences=[]
    for i, sent in enumerate(data_file):

        if args.max_pairs!=0 and i==args.max_pairs:
            break

        # firs we need the subword model to turn sentence into subwords
        # TODO
        if args.subword_model:
            sent=" ".join(sp.EncodeAsPieces(sent.strip().lower()))

        # now we can use vocabulary (whitespacevocab) to turn pieces into numbers the encoder understands
        if args.prefix=="fi":
            sent_vectorized, _=vs.sentence_vectorizer(sent, "")
        elif args.prefix=="en":
            _, sent_vectorized=vs.sentence_vectorizer("", sent)
        else:
            assert False, "Cannot decide vectorizer from prefix!" # TODO fix this
            
        sentences.append(sent_vectorized)

        if len(sentences)==batch:
            v_data=pad_sequences(np.array(sentences), maxlen=max_seq_len, padding='post', truncating='post')
            yield v_data
            sentences=[]
    if sentences:
        yield pad_sequences(np.array(sentences), maxlen=max_seq_len, padding='post', truncating='post')



def vectorize(args):

    # read and create files
    input_file=gzip.open(args.data_dir+"/{prefix}_len{N}.txt.gz".format(prefix=args.prefix, N=args.length),"rt",encoding="utf-8")
    output_file=open(args.data_dir+"/{prefix}_len{N}.npy".format(prefix=args.prefix, N=args.length),"wb")
    

    minibatch_size=100 

    #Read vocabularies
    vs=data_dense.WhitespaceSeparatedVocab()
    vs.load_vocabularies(args.vocabulary+"-vocab.json") 
    vs.trainable=False
    
    # load model
    encoder_model=load_model(args)
    max_seq_len=50 # TODO
    output_size=1024 # TODO

    

    # get vectors
    # for loop over minibatches
    counter=0
    for i,vbatch in enumerate(data_vectorizer(minibatch_size, max_seq_len, vs, input_file, args)):
        predictions=encoder_model.predict(vbatch) # shape = (minibatch_size,gru_width)
        # loop over items in minibatch
        for j,vector in enumerate(predictions): # does it matter that the last batch is smaller?
            norm_vector=vector/np.linalg.norm(vector)
            norm_vector.astype(np.float32).tofile(output_file)
            
            counter+=1
            if counter%100000==0:
                print("Vectorized {c} sentences".format(c=counter),file=sys.stderr,flush=True)
                
    input_file.close()
    output_file.close()

    print("Vectorized {c} sentences".format(c=counter),file=sys.stderr,flush=True)


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--prefix', type=str, help='Model to turn text into subword units (empty for no subwords)')
    g.add_argument('--subword_model', type=str, help='Model to turn text into subword units (empty for no subwords)')
    g.add_argument('--data_dir', type=str, help='Directory where text files are and where vectors should be written.')
    g.add_argument('-l', '--length', type=str, help='Sentence length, tells us which files to read')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give max pairs of sentences to read, zero for all, default={n}'.format(n=1000))
    
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None or args.length==None or args.data_dir==None or args.prefix==None:
        parser.print_help()
        sys.exit(1)

    number=str(args.max_pairs) if args.max_pairs!=0 else "all"
    print("Vectorizing",number,"sentences",file=sys.stderr)

    vectorize(args)

    print("Done.")

#    for s in iter_wrapper("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",max_sent=10000000):
#        pass




