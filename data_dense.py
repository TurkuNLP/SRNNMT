### PYTHON 3 code ###

"""
This module reads the data in...
"""

import itertools
import collections
import numpy as np
import sys
import json
import os.path
import random

import logging
logging.basicConfig(level=0)
from keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm

### DATA ITERATORS (return always sentences) ###

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


def infinite_iterator(source, target, monolingual_source, monolingual_target, max_iterations=None, max_pairs=None):
        " return one monolingual source, one monolingual tagret, and one parallel pair in each yield "
        parallel_data=list(iter_parallel_data(source,target,max_pairs))
        monolingual_source_iter=monolingual_iterator(monolingual_source)
        monolingual_target_iter=monolingual_iterator(monolingual_target)
        counter=0
        while True:
            for src_sent,trg_sent in parallel_data:
                yield next(monolingual_source_iter), next(monolingual_target_iter), (src_sent,trg_sent)
            counter+=1
            if max_iterations is not None and counter==max_iterations:
                break


### VOCABULARIES ###

class VocabularyChar(object):

    def __init__(self):
        # initialize, must run build after this
        self.source_vocab={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}  
        self.target_vocab={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3} 
        self.trainable=True    #If false, it will use <UNK>

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict["<UNK>"])


    def save_vocabularies(self, f_name):
        with open(f_name,"wt") as f:
            json.dump((self.source_vocab, self.target_vocab),f)

    def load_vocabularies(self, f_name):
        with open(f_name,"rt") as f:
            self.source_vocab, self.target_vocab = json.load(f)


    def build(self, voc_fname, source, target, monolingual_source, monolingual_target, force_rebuild=False):
        counter=0
        if force_rebuild or not os.path.exists(voc_fname):
            logging.info("Making one pass to gather vocabulary")
            for _,_,(sent_src,sent_target) in infinite_iterator(source, target, monolingual_source, monolingual_target, max_iterations=1, max_pairs=1000000):
                counter+=1
                if counter%10000==0:
                    logging.info("Seen {} sentence pairs...".format(counter))
                for char in sent_src:
                    self.get_id(char,self.source_vocab)
                for char in sent_target:
                    self.get_id(char,self.target_vocab)
            logging.info("Saving new vocabularies to "+voc_fname)
            self.save_vocabularies(voc_fname)
            self.source_vocab_size=len(self.source_vocab)
            self.target_vocab_size=len(self.target_vocab)
        else: # Load existing
            logging.info("Loading vocabularies from "+voc_fname)
            self.source_vocab,self.target_vocab=self.load_vocabularies(voc_fname)
            self.source_vocab_size=len(self.source_vocab)
            self.target_vocab_size=len(self.target_vocab)


    def sentence_vectorizer(self, source, target):
        """ character segmenter """
        s_sent=[self.get_id("<BOS>",self.source_vocab)]
        t_sent=[self.get_id("<BOS>",self.target_vocab)]
        for char in source:
            s_sent.append(self.get_id(char,self.source_vocab))
        for char in target:
            t_sent.append(self.get_id(char,self.target_vocab))
        s_sent.append(self.get_id("<EOS>",self.source_vocab))
        t_sent.append(self.get_id("<EOS>",self.target_vocab))
        return s_sent, t_sent


    def inversed_vectorizer_source(self, source):
        if not self.inversed_source:
            self.inversed_source={v:k for k,v in self.source_vocab.items()}
        return [self.inversed_source[i] for i in source]

    def inversed_vectorizer_target(self, target):
        if not self.inversed_target:
            self.inversed_target={v:k for k,v in self.target_vocab.items()}
        return [self.inversed_target[i] for i in target]



class WhitespaceSeparatedVocab(object):

    def __init__(self):
        # initialize, must run build after this
        self.source_vocab={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}  
        self.target_vocab={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3} 
        self.trainable=True    #If false, it will use <UNK>
        self.inversed_source=None
        self.inversed_target=None

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict["<UNK>"])


    def save_vocabularies(self, f_name):
        with open(f_name,"wt") as f:
            json.dump((self.source_vocab, self.target_vocab),f)

    def load_vocabularies(self, f_name):
        with open(f_name,"rt") as f:
            self.source_vocab, self.target_vocab = json.load(f)


    def build(self, voc_fname, source, target, monolingual_source, monolingual_target, force_rebuild=False):
        counter=0
        if force_rebuild or not os.path.exists(voc_fname):
            logging.info("Making one pass to gather vocabulary")
            for _,_,(sent_src,sent_target) in infinite_iterator(source, target, monolingual_source, monolingual_target, max_iterations=1, max_pairs=1000000):
                counter+=1
                if counter%10000==0:
                    logging.info("Seen {} sentence pairs...".format(counter))
                for item in sent_src.split(" "):
                    self.get_id(item,self.source_vocab)
                for item in sent_target.split(" "):
                    self.get_id(item,self.target_vocab)
            logging.info("Saving new vocabularies to "+voc_fname)
            self.save_vocabularies(voc_fname)
            self.source_vocab_size=len(self.source_vocab)
            self.target_vocab_size=len(self.target_vocab)
        else: # Load existing
            logging.info("Loading vocabularies from "+voc_fname)
            self.load_vocabularies(voc_fname)
            self.source_vocab_size=len(self.source_vocab)
            self.target_vocab_size=len(self.target_vocab)


    def sentence_vectorizer(self, source, target):
        """ character segmenter """
        s_sent=[self.get_id("<BOS>",self.source_vocab)]
        t_sent=[self.get_id("<BOS>",self.target_vocab)]
        for item in source.split(" "):
            s_sent.append(self.get_id(item,self.source_vocab))
        for item in target.split(" "):
            t_sent.append(self.get_id(item,self.target_vocab))
        s_sent.append(self.get_id("<EOS>",self.source_vocab))
        t_sent.append(self.get_id("<EOS>",self.target_vocab))
        return s_sent, t_sent


    def inversed_vectorizer_source(self, source):
        if not self.inversed_source:
            self.inversed_source={v:k for k,v in self.source_vocab.items()}
        return [self.inversed_source[i] for i in source]

    def inversed_vectorizer_target(self, target):
        if not self.inversed_target:
            self.inversed_target={v:k for k,v in self.target_vocab.items()}
        return [self.inversed_target[i] for i in target]



class VocabularySubWord(object):

    def __init__(self):
        # initialize, must run build after this
        self.source_vocab={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}  
        self.target_vocab={"<MASK>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3} 
        self.trainable=True    #If false, it will use <UNK>

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict["<UNK>"])


    def save_vocabularies(self, f_name):
        with open(f_name,"wt") as f:
            json.dump((self.source_vocab, self.target_vocab),f)

    def load_vocabularies(self, f_name):
        with open(f_name,"rt") as f:
            self.source_vocab, self.target_vocab = json.load(f)

    def train_subwords(self, fname, model_name):
        spm.SentencePieceTrainer.Train("--input={f} --model_prefix={m} --vocab_size=20000".format(f=fname, m=model_name))

    def load_subwords(self, model_name):
        self.subword_model_source = spm.SentencePieceProcessor()
        self.subword_model_source.Load(model_name+"-src.model")
        self.source_vocab_size=self.subword_model_source.GetPieceSize()
        self.subword_model_target = spm.SentencePieceProcessor()
        self.subword_model_target.Load(model_name+"-trg.model")
        self.target_vocab_size=self.subword_model_target.GetPieceSize()

    def build(self, voc_fname, source, target, monolingual_source, monolingual_target, force_rebuild=False):
        counter=0
        if force_rebuild or not os.path.exists(voc_fname+"-src.model"):
            logging.info("Build subword model")
            self.train_subwords("data/subwords/fi.input",voc_fname+"-src")
            self.train_subwords("data/subwords/en.input",voc_fname+"-trg")
            self.load_subwords(voc_fname)
        else: # Load existing
            logging.info("Loading subword models from "+voc_fname)
            self.load_subwords(voc_fname)


    def sentence_vectorizer(self, source, target):
        """ character segmenter """
        source="<BOS>"+source+"<EOS>"
        target="<BOS>"+target+"<EOS>"
        s=self.subword_model_source.EncodeAsIds(source)
        t=self.subword_model_source.EncodeAsIds(target)
        return s, t

    def inversed_vectorizer_source(self, data):
        return self.subword_model_source.DecodeIds(data)

    def inversed_vectorizer_target(self, data):
        return self.subword_model_target.DecodeIds(data)

def fill_batch(minibatch_size,max_seq_len,vs,data_iterator):
    
    src_sentences=[]
    trg_sentences=[]
    src_sentences_mono=[]
    trg_sentences_mono=[]
    for mono_src, mono_trg, (sent_src,sent_trg) in data_iterator:
        # parallel data
        s,t=vs.sentence_vectorizer(sent_src,sent_trg)
        src_sentences.append(s)
        trg_sentences.append(t)
        # monolingual data
        s,t=vs.sentence_vectorizer(mono_src,mono_trg)
        src_sentences_mono.append(s)
        trg_sentences_mono.append(t)
        
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
