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
import lwvlib

import logging
logging.basicConfig(level=0)

# This named tuple holds the matrices which make one minibatch
class Matrices:
    
    def __init__(self,minibatch_size,max_sent_len,ngrams):
#        self.source_ngrams={} #{ N -> matrix}
#        self.target_ngrams={} #{ N -> matrix}
#        for N in ngrams:
#            self.source_ngrams[N]=np.zeros((minibatch_size,max_sent_len),np.int)
#            self.target_ngrams[N]=np.zeros((minibatch_size,max_sent_len),np.int)
#        self.src_len=np.zeros((minibatch_size,1),np.int)
#        self.trg_len=np.zeros((minibatch_size,1),np.int)
        self.targets=np.zeros((minibatch_size,1),np.float32)
        self.source_words=np.zeros((minibatch_size,max_sent_len),np.int)
        self.target_words=np.zeros((minibatch_size,max_sent_len),np.int)
        self.matrix_dict={}
#        for N in ngrams:
#            self.matrix_dict["source_ngrams_{}".format(N)]=self.source_ngrams[N] #we need a string identifier
#            self.matrix_dict["target_ngrams_{}".format(N)]=self.target_ngrams[N] #we need a string identifier
#        self.matrix_dict["src_len"]=self.src_len
#        self.matrix_dict["trg_len"]=self.trg_len
        self.matrix_dict["source_words"]=self.source_words
        self.matrix_dict["target_words"]=self.target_words

    def wipe(self):
        assert False #I think this one I don't use anymore
        for m in self.source_chars, self.target_chars, self.targets:
            m.fill(0)

class Vocabularies(object):

    def __init__(self,src_vfile,trg_vfile):
#        self.source_ngrams={}  # {order -> {"<MASK>":0,"<UNK>":1}} # source language ngrams
#        self.target_ngrams={}  # {order -> {"<MASK>":0,"<UNK>":1}} # target language characters
#        for N in ngrams:
#            self.source_ngrams[N]={"<MASK>":0,"<UNK>":1} # source language ngrams
#            self.target_ngrams[N]={"<MASK>":0,"<UNK>":1} # # target language characters
        self.trainable=True    #If false, it will use <UNK>
#        self.source_words={"<MASK>":0,"<UNK>":1}
#        self.target_words={"<MASK>":0,"<UNK>":1}
        self.source_words,self.src_vectors=self.read_vocabs(src_vfile)
        self.target_words,self.trg_vectors=self.read_vocabs(trg_vfile)
        print(self.src_vectors.shape)
        print(self.trg_vectors.shape)
        
        
    def read_vocabs(self,vmodel,vsize=100000):
        model=lwvlib.load(vmodel,vsize,vsize)
        w2idx={"<MASK>":0,"<UNK>":1}
        for i,w in enumerate(model.words[:vsize]):
            w2idx[w]=i+2
#        print(len(w2idx))
        vectors=np.zeros((vsize+2,model.vectors.shape[1]),np.float)
        for i,vec in enumerate(model.vectors):
            vectors[i+2]=vec
        # vector for unk
        vectors[1]=vectors[np.random.randint(2,len(model.words))] # take random # TODO: fix!
        return w2idx,vectors
        
        

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict["<UNK>"])


def iter_data(training_source,training_target):
    with open(training_source) as src, open(training_target) as trg:
        for src_line, trg_line in zip(src,trg):
            if 5<=len(src_line.strip().split())<=30 and 5<=len(trg_line.strip().split())<=30: # sentence length must be between 5 and 25 tokens
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

def ngram_iterator(src,N,max_sentence_len):
    if N==1:
        yield from src[:max_sentence_len]
    else:
        for i in range(len(src)-N+1):
            if i>=max_sentence_len:
                break
            yield src[i:i+N]
    
            
def read_vocabularies(training_source,training_target,force_rebuild,ngrams):
    #ngrams -> (1,2,3)... iterable of Ns
    voc_fname=training_source+"-vocabularies.pickle"
    if force_rebuild or not os.path.exists(voc_fname):
        #make sure no feature has 0 index
        logging.info("Making one pass to gather vocabulary")
        vs=Vocabularies(ngrams)
        for (sent_src,sent_target),_ in InfiniteDataIterator(training_source,training_target,max_iterations=1): #Make a single pass: # (source_sentence, target_sentence)
#            for N in ngrams:
#                for ngram in ngram_iterator(sent_src,N,len(sent_src)):
#                    vs.get_id(ngram,vs.source_ngrams[N])
#                for ngram in ngram_iterator(sent_target,N,len(sent_target)):
#                    vs.get_id(ngram,vs.target_ngrams[N])
            for token in sent_src.strip().split():
                vs.get_id(token,vs.source_words)
            for token in sent_target.strip().split():
                vs.get_id(token,vs.target_words)
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

def fill_batch(minibatch_size,max_sent_len,vs,data_iterator,ngrams):
    """ Iterates over the data_iterator and fills the index matrices with fresh data
        ms = matrices, vs = vocabularies
    """


    ms=Matrices(minibatch_size,max_sent_len,ngrams)
    batchsize,max_sentence_len=ms.source_words.shape #just pick any one of these really
    row=0
    for (sent_src,sent_target),target in data_iterator:
#        for N in ngrams:
#            for j,ngram in enumerate(ngram_iterator(sent_src,N,max_sent_len)):
#                ms.source_ngrams[N][row,j]=vs.get_id(ngram,vs.source_ngrams[N])
#            for j,ngram in enumerate(ngram_iterator(sent_target,N,max_sent_len)):
#                ms.target_ngrams[N][row,j]=vs.get_id(ngram,vs.target_ngrams[N])
        for x,token in enumerate(sent_src.strip().split()):
            ms.source_words[row,x]=vs.get_id(token,vs.source_words)
        for x,token in enumerate(sent_target.strip().split()):
            ms.target_words[row,x]=vs.get_id(token,vs.target_words)
#        ms.src_len[row]=len(sent_src.strip().split())
#        ms.trg_len[row]=len(sent_target.strip().split())
        ms.targets[row]=target
        row+=1
        if row==batchsize:
#            print(ms.matrix_dict, ms.targets)
            yield ms.matrix_dict, ms.targets
            row=0
            ms=Matrices(minibatch_size,max_sent_len,ngrams)


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
