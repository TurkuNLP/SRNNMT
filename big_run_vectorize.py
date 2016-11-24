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
import pickle
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import hashlib
import array

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
    total_count=0
    yielded=0
    fnames=glob.glob(dirname+"/*.gz")
    fnames.sort()
#    print(fnames)
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr,flush=True)
        for comm, sent in cu.read_conllu(gzip.open(fname,"rt",encoding="utf-8")):
            total_count+=1
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
    print("Fin parsebank:",total_count,file=sys.stderr)
    print("Vectorized:",count,file=sys.stderr)

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
    total_count=0
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr,flush=True)
        for sent in sent_reader(gzip.open(fname,"rt",encoding="utf-8")):
            total_count+=1
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
    print("Eng parsebank:",total_count,file=sys.stderr)
    print("Vectorized:",count,file=sys.stderr)  
    
    
def build_sparse_matrices(outdir,fname,translation_dictionary):
    # return sparse, and translated_sparse
    print("Buildinf sparse from",fname)
    f=gzip.open(outdir+"/"+fname,"rt",encoding="utf-8")
    sparse_size=1000000
    normalizer=np.zeros(sparse_size,dtype=np.float32)
    row=array.array('i')
    col=array.array('i')
    transl_row=array.array('i')
    transl_col=array.array('i')
    counter=0
    for i,sent in enumerate(f):
        words=set(sent.strip().lower().split())
        normalizer[i]=np.float32(len(words))
        for word in words:
            if word not in translation_dictionary:
                continue
            h=int(hashlib.sha224(word.encode("utf-8")).hexdigest(), 16)%sparse_size
            row.append(i)
            col.append(h)
            for translation in translation_dictionary[word]:
                h=int(hashlib.sha224(translation.encode("utf-8")).hexdigest(), 16)%sparse_size
                transl_row.append(i)
                transl_col.append(h)
        counter+=1
        if i!=0 and i%10000==0:
            print(i)
    normalizer=normalizer[:counter]
    sparse=coo_matrix((np.ones(len(row),dtype=np.float32),(np.frombuffer(row,dtype=np.int32),np.frombuffer(col,dtype=np.int32))),shape=(counter,sparse_size),dtype=np.float32)
    translated_sparse=coo_matrix((np.ones(len(transl_row),dtype=np.float32),(np.frombuffer(transl_row,dtype=np.int32),np.frombuffer(transl_col,dtype=np.int32))),shape=(counter,sparse_size),dtype=np.float32)
    if fname.startswith("fi"):
        sparse=sparse.tocsr()
        translated_sparse=translated_sparse.tocsr()
    elif fname.startswith("en"):
        sparse=sparse.tocsc()
        translated_sparse=translated_sparse.tocsc()
    f.close()
    with open(outdir+"/"+fname.replace(".txt.gz",".sparse.pickle"), "wb") as f:
        pickle.dump(sparse,f)
    with open(outdir+"/"+fname.replace(".txt.gz",".translated_sparse.pickle"), "wb") as f:
        pickle.dump(translated_sparse,f)
    with open(outdir+"/"+fname.replace(".txt.gz",".normalizer.pickle"), "wb") as f:
        pickle.dump(normalizer,f)
    # saved, return nothing
  
  
            
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
    outdir="vdata_bible"    
    file_dict={}
    for i in range(min_len,max_len+1): # C is here 0
        file_dict["fi_sent_len{N}".format(N=i)]=gzip.open(outdir+"/fi_len{N}_{C}.txt.gz".format(N=i,C=0),"wt",encoding="utf-8")
        file_dict["fi_vec_len{N}".format(N=i)]=open(outdir+"/fi_len{N}_{C}.npy".format(N=i,C=0),"wb")
        file_dict["fi_count_len{N}".format(N=i)]=0
        file_dict["en_sent_len{N}".format(N=i)]=gzip.open(outdir+"/en_len{N}_{C}.txt.gz".format(N=i,C=0),"wt",encoding="utf-8")
        file_dict["en_vec_len{N}".format(N=i)]=open(outdir+"/en_len{N}_{C}.npy".format(N=i,C=0),"wb")
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
    
    # read translation dictionaries
    with open("f2e_dictionary.pickle", "rb") as f:
        f2e_dictionary=pickle.load(f)
    with open("e2f_dictionary.pickle", "rb") as f:
        e2f_dictionary=pickle.load(f)
        
#    f2e_dictionary=build_dictionary("lex.f2e","uniq.train.tokens.fi.100K")
#    e2f_dictionary=build_dictionary("lex.e2f","uniq.train.tokens.en.100K")
    

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
                    # now create sparse matrices from previous batch...
                    fname="fi_len{N}_{C}.txt.gz".format(N=fi_len,C=file_dict["fi_count_len{N}".format(N=fi_len)]-1000000)
                    print(fname,file=sys.stderr,flush=True)
                    build_sparse_matrices(outdir,fname,f2e_dictionary)
                    
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
                    
                    # now create sparse matrices from previous batch...
                    fname="en_len{N}_{C}.txt.gz".format(N=en_len,C=file_dict["en_count_len{N}".format(N=en_len)]-1000000)
                    print(fname,file=sys.stderr,flush=True)
                    build_sparse_matrices(outdir,fname,e2f_dictionary)
            
            counter+=1
            if counter%100000==0:
                print("Vectorized {c} sentence pairs".format(c=counter),file=sys.stderr,flush=True)
                

    for key,value in file_dict.items():
        if "count" in key:
            continue
        elif "sent" in key: # create last sparse matrices
            if key.startswith("fi"):
                d=f2e_dictionary
            else:
                d=e2f_dictionary
            fname=value.name.split("/",1)[1]
            value.close()
            build_sparse_matrices(outdir,fname,d)
        else:
            value.close()




if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give max pairs of sentences to read, zero for all, default={n}'.format(n=1000))
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None:
        parser.print_help()
        sys.exit(1)


    vectorize(args.vocabulary,args.model,"/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",args.max_pairs)

#    for s in iter_wrapper("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",max_sent=10000000):
#        pass




