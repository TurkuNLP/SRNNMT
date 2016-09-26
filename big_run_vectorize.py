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
import conllutil3 as cu
import html
import glob
import itertools

import data_dense
import re

from test import load_model
from dictionary_baseline import build_dictionary


min_len=5
max_len=30

filter_regex=re.compile("[A-Za-zÅÄÖåäö]")

def good_text(sent):
    l=len(re.sub("\s+","", sent))
    if len(filter_regex.findall(sent))/l>0.8:
        return True
    else:
        return False

def read_fin_parsebank(dirname,max_sent=10000):
    fnames=glob.glob(dirname+"/*.gz")
    fnames.sort()
#    print(fnames)
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr)
        for comm, sent in cu.read_conllu(gzip.open(fname,"rt",encoding="utf-8")):
            if min_len<=len(sent)<=max_len:
                txt=" ".join(line[cu.FORM] for line in sent)
                if not good_text(txt):
                    continue
                if txt in uniq:
                    continue
                yield txt
                uniq.add(txt)
                counter+=1
#            if max_sent!=0 and counter==max_sent:
#                break
                

def sent_reader(f):
    words=[]
    for line in f:
        line=line.strip()
        if line=="</s>": # end of sentence
            if words:
                yield words
                words=[]
        cols=line.split("\t")
        if len(cols)==1:
            continue
        words.append(cols[0])
        

def read_eng_parsebank(dirname,max_sent=10000):
    fnames=glob.glob(dirname+"/*.xml.gz")
    fnames.sort()
#    print(fnames)
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr)
        for sent in sent_reader(gzip.open(fname,"rt",encoding="utf-8")):
            if min_len<=len(sent)<=max_len:
                txt=html.unescape(" ".join(sent)) # cow corpus: &apos;
                if not good_text(txt):
                    continue
                if txt in uniq:
                    continue
                yield txt
                uniq.add(txt)
                counter+=1
#            if max_sent!=0 and counter==max_sent:
#                break
            
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
        

def iter_wrapper(src_dirname,trg_dirname,max_sent=10000):
    counter=0
    for fin_sent,eng_sent in itertools.zip_longest(read_fin_parsebank(src_dirname,max_sent=max_sent),read_eng_parsebank(trg_dirname,max_sent=max_sent),fillvalue="#None#"): # shorter padded with 'None'
        yield (fin_sent,eng_sent),1.0
        counter+=1
        if max_sent!=0 and counter==max_sent:
            break
        

def vectorize(voc_name,mname,src_fname,trg_fname,max_pairs):
    # create files
    outdir="vdata_final"
    file_dict={}
    for i in range(min_len,max_len+1):
        file_dict["fi_sent_len{N}".format(N=i)]=gzip.open(outdir+"/fi_len{N}.txt.gz".format(N=i),"wt",encoding="utf-8")
        file_dict["fi_vec_len{N}".format(N=i)]=open(outdir+"/fi_len{N}.npy".format(N=i),"wb")
        file_dict["fi_count_len{N}".format(N=i)]=0
        file_dict["en_sent_len{N}".format(N=i)]=gzip.open(outdir+"/en_len{N}.txt.gz".format(N=i),"wt",encoding="utf-8")
        file_dict["en_vec_len{N}".format(N=i)]=open(outdir+"/en_len{N}.npy".format(N=i),"wb")
        file_dict["en_count_len{N}".format(N=i)]=0
    

    minibatch_size=100 
    ngrams=(4,) # TODO: read this from somewhere

    #Read vocabularies
    vs=data_dense.read_vocabularies(voc_name,"xxx","xxx",False,ngrams) 
    vs.trainable=False
    
    # load model
    trained_model=load_model(mname)
    output_size=trained_model.get_layer('source_dense').output_shape[1]
    max_sent_len=trained_model.get_layer('source_ngrams_{n}'.format(n=ngrams[0])).output_shape[1]
    print(output_size,max_sent_len,file=sys.stderr)
    
    # build matrices
    ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
    

    # get vectors
    # for loop over minibatches
    counter=0    
    for i,(mx,targets,src_data,trg_data) in enumerate(fill_batch(minibatch_size,max_sent_len,vs,iter_wrapper(src_fname,trg_fname,max_sent=max_pairs),ngrams)):
        src,trg=trained_model.predict(mx) # shape = (minibatch_size,gru_width)
        # loop over items in minibatch
        for j,(src_v,trg_v) in enumerate(zip(src,trg)):
            norm_src=src_v/np.linalg.norm(src_v)
            norm_trg=trg_v/np.linalg.norm(trg_v)
            if src_data[j]!="#None#":
                fi_len=len(src_data[j].split())
                norm_src.astype(np.float32).tofile(file_dict["fi_vec_len{N}".format(N=fi_len)])
                print(src_data[j],file=file_dict["fi_sent_len{N}".format(N=fi_len)])
                file_dict["fi_count_len{N}".format(N=fi_len)]+=1
                if file_dict["fi_count_len{N}".format(N=fi_len)]%1000000==0:
                    file_dict["fi_vec_len{N}".format(N=fi_len)].close()
                    file_dict["fi_vec_len{N}".format(N=fi_len)]=open(outdir+"/fi_len{N}_{C}.npy".format(N=fi_len,C=file_dict["fi_count_len{N}".format(N=fi_len)]),"wb")
                    file_dict["fi_sent_len{N}".format(N=fi_len)].close()
                    file_dict["fi_sent_len{N}".format(N=fi_len)]=gzip.open(outdir+"/fi_len{N}_{C}.txt.gz".format(N=fi_len,C=file_dict["fi_count_len{N}".format(N=fi_len)]),"wt",encoding="utf-8")
                    
            if trg_data[j]!="#None#":
                en_len=len(trg_data[j].split())
                norm_trg.astype(np.float32).tofile(file_dict["en_vec_len{N}".format(N=en_len)])
                print(trg_data[j],file=file_dict["en_sent_len{N}".format(N=en_len)])
                file_dict["en_count_len{N}".format(N=en_len)]+=1
                if file_dict["en_count_len{N}".format(N=en_len)]%1000000==0:
                    file_dict["en_vec_len{N}".format(N=en_len)].close()
                    file_dict["en_vec_len{N}".format(N=en_len)]=open(outdir+"/en_len{N}_{C}.npy".format(N=en_len,C=file_dict["en_count_len{N}".format(N=en_len)]),"wb")
                    file_dict["en_sent_len{N}".format(N=en_len)].close()
                    file_dict["en_sent_len{N}".format(N=en_len)]=gzip.open(outdir+"/en_len{N}_{C}.txt.gz".format(N=en_len,C=file_dict["en_count_len{N}".format(N=en_len)]),"wt",encoding="utf-8")
            
            counter+=1
            if counter%100000==0:
                print("Vectorized {c} sentence pairs".format(c=counter))

    for key,value in file_dict.items():
        if "count" in key:
            continue
        value.close()




if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, zero for all, default={n}'.format(n=1000))
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None:
        parser.print_help()
        sys.exit(1)


    vectorize(args.vocabulary,args.model,"/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",args.max_pairs)

#    for s in iter_wrapper("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",max_sent=10000000):
#        pass




