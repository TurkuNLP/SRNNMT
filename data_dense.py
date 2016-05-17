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

# This named tuple holds the matrices which make one minibatch
# We will not be making new ones for every minibatch, we'll just wipe the existing ones
class Matrices:
    
    def __init__(self,minibatch_size,max_sent_len):
        self.source_chars=np.zeros((minibatch_size,max_sent_len),np.int)
        self.target_chars=np.zeros((minibatch_size,max_sent_len),np.int)
        self.targets=np.zeros((minibatch_size,),np.float32)
        self.matrix_dict={"source_chars":self.source_chars,"target_chars":self.target_chars}

    def wipe(self):
        for m in self.source_chars, self.target_chars, self.targets:
            m.fill(0)

class Vocabularies(object):

    def __init__(self):
        self.source_chars={"<MASK>":0,"<UNK>":1} # source language characters
        self.target_chars={"<MASK>":0,"<UNK>":1} # target language characters
        self.trainable=True #If false, it will use <UNK>

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict["<UNK>"])


def iter_data(training_source,training_target):
    with open(training_source) as src, open(training_target) as trg:
        for src_line, trg_line in zip(src,trg):
            yield src_line, trg_line

class InfiniteDataIterator:

    def __init__(self,training_source,training_target,max_iterations=None):
        self.training_source=training_source
        self.training_target=training_target
        self.max_iterations=max_iterations
        self.data=list(iter_data(self.training_source,self.training_target)) #must memorize
                
    def __iter__(self):
        """
        Returns randomized training pairs as ((source_sent,target_sent),-1/+1)
        """
        positive_indices=list(zip(range(len(self.data)),range(len(self.data)))) #[(0,0),(1,1),(2,2)...]  #indices of source,target
        counter=0
        while True:
            indices=list(range(len(self.data)))
            random.shuffle(indices) #shuffled indices
            negative_indices=list(enumerate(indices)) #[(0,326543),(1,96457),...]
            all_examples=positive_indices+negative_indices
            random.shuffle(all_examples) 
            for src_idx,trg_idx in all_examples: #index where I should take the source sentence, index where I should take the target sentence
                if src_idx==trg_idx: #same -> positive example
                    yield self.data[src_idx],1.0
                else: #different -> negative example
                    yield (self.data[src_idx][0],self.data[trg_idx][1]),-1.0
            counter+=1
            if self.max_iterations is not None and counter==self.max_iterations:
                break

def read_vocabularies(training_source,training_target,force_rebuild):
    voc_fname=training_source+"-vocabularies.pickle"
    if force_rebuild or not os.path.exists(voc_fname):
        #make sure no feature has 0 index
        logging.info("Making one pass to gather vocabulary")
        vs=Vocabularies()
        for (sent_src,sent_target),_ in InfiniteDataIterator(training_source,training_target,max_iterations=1): #Make a single pass: # (source_sentence, target_sentence)
            for c in itertools.filterfalse(str.isspace,sent_src):
                vs.get_id(c,vs.source_chars)
            for c in itertools.filterfalse(str.isspace,sent_target):
                vs.get_id(c,vs.target_chars)
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


def fill_batch(ms,vs,data_iterator):
    """ Iterates over the data_iterator and fills the index matrices with fresh data
        ms = matrices, vs = vocabularies
    """

    #matrix_dict=dict(zip(ms._fields,ms)) #the named tuple as dict, what we return
    batchsize,max_sentence_len=ms.source_chars.shape
    row=0
    for (sent_src,sent_target),target in data_iterator:
        for j,c in enumerate(itertools.islice(itertools.filterfalse(str.isspace,sent_src),max_sentence_len)):
            ms.source_chars[row,j]=vs.get_id(c,vs.source_chars)
        for j,c in enumerate(itertools.islice(itertools.filterfalse(str.isspace,sent_target),max_sentence_len)):
            ms.target_chars[row,j]=vs.get_id(c,vs.target_chars)
        ms.targets[row]=target
        row+=1
        if row==batchsize:
            yield ms.matrix_dict, ms.targets
            row=0
            ms.wipe()


if __name__=="__main__":
    vs=read_vocabularies("data/JRC-Acquis.en-fi.fi","data/JRC-Acquis.en-fi.en",False)
    vs.trainable=False
    ms=Matrices(100,500) #minibatchsize,max_sent_len
    raw_data=infinite_iter_data("data/JRC-Acquis.en-fi.fi","data/JRC-Acquis.en-fi.en")
    for minibatch,targets in fill_batch(ms,vs,raw_data):
        print(minibatch["source_chars"])
        print(minibatch["target_chars"])
        print(targets)
        break
