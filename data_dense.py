### PYTHON 3 code ###

"""
This module reads the data in...
"""

import itertools
import collections
import numpy as np
import sys
import pickle
import os.path
import random

import logging
logging.basicConfig(level=0)
from keras.preprocessing.sequence import pad_sequences


class Vocabularies(object):

    def __init__(self):
        self.source_char={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}  
        self.target_char={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3} 
        self.trainable=True    #If false, it will use <UNK>

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict["<UNK>"])


def iter_parallel_data(training_source,training_target,max_pairs=None):
    counter=0
    with open(training_source) as src, open(training_target) as trg:
        for src_line, trg_line in zip(src,trg):
            src_line,trg_line=src_line.strip(),trg_line.strip()
            if not src_line or not trg_line: # remove possible empty lines
                continue
            yield src_line, trg_line
            counter+=1
            if max_pairs is not None and max_pairs>0 and counter>=max_pairs:
                break

def monolingual_iterator(fname):
    while True:
        if fname.endswith(".gz"):
            import gzip
            f=gzip.open(fname,"rt",encoding="utf-8")
        else:
            f=open(fname,"rt",encoding="utf-8")
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield line
        f.close()

class InfiniteDataIterator:
    """ Class to handle the data. """

    def __init__(self,training_source,training_target,monolingual_source,monolingual_target,max_iterations=None,max_pairs=None):
        self.training_source=training_source
        self.training_target=training_target
        self.monolingual_source=monolingual_iterator(monolingual_source)
        self.monolingual_target=monolingual_iterator(monolingual_target)
        self.max_iterations=max_iterations
        self.max_pairs=max_pairs
        self.parallel_data=list(iter_parallel_data(self.training_source,self.training_target,self.max_pairs)) 
                
    def __iter__(self):
        " return one monolingual source, one monolingual tagret, and one parallel pair in each yield "
        counter=0
        while True:
            for src_sent,trg_sent in self.parallel_data:
                yield next(self.monolingual_source), next(self.monolingual_target), (src_sent,trg_sent)
            counter+=1
            if self.max_iterations is not None and counter==self.max_iterations:
                break


            

def ngram_iterator(src,N,max_sentence_len):
    if N==1:
        yield from src[:max_sentence_len]
    else:
        for i in range(len(src)-N+1):
            if i>=max_sentence_len:
                break
            yield src[i:i+N]
    
            
def read_vocabularies(voc_fname,training_source,training_target,monolingual_source,monolingual_target,force_rebuild=False):
    counter=0
    if force_rebuild or not os.path.exists(voc_fname):
        #make sure no feature has 0 index
        logging.info("Making one pass to gather vocabulary")
        vs=Vocabularies()
        for _,_,(sent_src,sent_target) in InfiniteDataIterator(training_source,training_target,monolingual_source, monolingual_target,max_iterations=1,max_pairs=100000): #Make a single pass: # (source_sentence, target_sentence)
            counter+=1
            if counter%10000==0:
                logging.info("Seen {} sentence pairs...".format(counter))
            for char in sent_src:
                vs.get_id(char,vs.source_char)
            for char in sent_target:
                vs.get_id(char,vs.target_char)
        logging.info("Saving new vocabularies to "+voc_fname)
        save_vocabularies(vs,voc_fname)
    else:
        logging.info("Loading vocabularies from "+voc_fname)
        vs=load_vocabularies(voc_fname)
    return vs


def save_vocabularies(vs,f_name):
    with open(f_name,"wb") as f:
        pickle.dump(vs,f,pickle.HIGHEST_PROTOCOL)

def load_vocabularies(f_name):
    with open(f_name,"rb") as f:
        return pickle.load(f)
        

def fill_batch(minibatch_size,max_seq_len,vs,data_iterator):
    
    src_sentences=[]
    trg_sentences=[]
    src_sentences_mono=[]
    trg_sentences_mono=[]
    for mono_src, mono_trg, (sent_src,sent_trg) in data_iterator:
        # parallel data
        s_sent=[vs.get_id("<BOS>",vs.source_char)]
        t_sent=[vs.get_id("<BOS>",vs.target_char)]
        for char in sent_src:
            s_sent.append(vs.get_id(char,vs.source_char))
        s_sent.append(vs.get_id("<EOS>",vs.source_char))
        for char in sent_trg:
            t_sent.append(vs.get_id(char,vs.target_char))
        t_sent.append(vs.get_id("<EOS>",vs.target_char))
        src_sentences.append(s_sent)
        trg_sentences.append(t_sent)

        # monolingual data
        s_sent=[vs.get_id("<BOS>",vs.source_char)]
        t_sent=[vs.get_id("<BOS>",vs.target_char)]
        for char in mono_src:
            s_sent.append(vs.get_id(char,vs.source_char))
        s_sent.append(vs.get_id("<EOS>",vs.source_char))
        for char in mono_trg:
            t_sent.append(vs.get_id(char,vs.target_char))
        t_sent.append(vs.get_id("<EOS>",vs.target_char))
        src_sentences_mono.append(s_sent)
        trg_sentences_mono.append(t_sent)
        
        if len(src_sentences)==minibatch_size:
            # parallel inputs
            src_data=pad_sequences(np.array(src_sentences), maxlen=max_seq_len, padding='post', truncating='post')
            trg_data=pad_sequences(np.array(trg_sentences), maxlen=max_seq_len, padding='post', truncating='post')
            #  parallel outputs
            src_out=np.expand_dims(src_data, -1)
            trg_out=np.expand_dims(trg_data, -1)
            # mono inputs
            src_data_mono=pad_sequences(np.array(src_sentences_mono), maxlen=max_seq_len, padding='post', truncating='post')
            trg_data_mono=pad_sequences(np.array(trg_sentences_mono), maxlen=max_seq_len, padding='post', truncating='post')
            yield (src_data_mono, np.expand_dims(src_data_mono, -1)), (trg_data_mono, np.expand_dims(trg_data_mono, -1)), (src_data, trg_data, src_out, trg_out) # monolingual src, monolingual trg, parallel
            src_sentences=[]
            trg_sentences=[]
            src_sentences_mono=[]
            trg_sentences_mono=[]
        


if __name__=="__main__":
    vs=read_vocabularies("data/JRC-Acquis.en-fi.fi","data/JRC-Acquis.en-fi.en",False,(3,4))
    vs.trainable=False
    ms=Matrices(100,500,(3,4)) #minibatchsize,max_sent_len,ngrams
    raw_data=InfiniteDataIterator("data/JRC-Acquis.en-fi.fi","data/JRC-Acquis.en-fi.en",1)
    for minibatch,targets in fill_batch(100,500,vs,raw_data,(3,4)):
        print(minibatch["source_ngrams_3"])
        print(minibatch["target_ngrams_3"])
        print(targets)
        break
