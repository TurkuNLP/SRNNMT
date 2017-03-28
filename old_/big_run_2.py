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

def read_fin_parsebank(fname,max_sent=10000):

#    sentences=[]
    counter=0
    uniq=set()
    for comm, sent in cu.read_conllu(gzip.open(fname,"rt",encoding="utf-8")):
        if min_len<=len(sent)<=max_len:
            txt=" ".join(line[cu.FORM] for line in sent)
            if not good_text(txt):
                continue
            if txt in uniq:
                continue
            yield txt
            uniq.add(txt)
#            sentences.append(txt)
            counter+=1
        if counter==max_sent:
            break
#    return sentences

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
        

def read_eng_parsebank(fname,max_sent=10000):
    counter=0
    uniq=set()
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
        if counter==max_sent:
            break
            
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
        


def iter_wrapper(src_fname,trg_fname,max_sent=10000):
    for fin_sent,eng_sent in zip(read_fin_parsebank(src_fname,max_sent=max_sent),read_eng_parsebank(trg_fname,max_sent=max_sent)):
        yield (fin_sent,eng_sent),1.0
        
#def iter_wrapper(src_fname,trg_fname,max_sent=1000):
#    count=0
#    for fin_sent,eng_sent in zip(open(src_fname),open(trg_fname)):
#        fin_sent=fin_sent.strip()
#        eng_sent=eng_sent.strip()
#        yield (fin_sent,eng_sent),1.0
#        count+=1
#        if count==max_sent:
#            break


def vectorize(voc_name,mname,src_fname,trg_fname,max_pairs):

    # create files
    file_dict={}
    for i in range(min_len,max_len+1):
        file_dict["fi_sent_len{N}".format(N=i)]=gzip.open("vdata_uniq/fi_sent_len{N}.txt.gz".format(N=i),"wt",encoding="utf-8")
        file_dict["fi_vec_len{N}".format(N=i)]=open("vdata_uniq/fi_vec_len{N}.npy".format(N=i),"wb")
        file_dict["en_sent_len{N}".format(N=i)]=gzip.open("vdata_uniq/en_sent_len{N}.txt.gz".format(N=i),"wt",encoding="utf-8")
        file_dict["en_vec_len{N}".format(N=i)]=open("vdata_uniq/en_vec_len{N}.npy".format(N=i),"wb")
    

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
            
            fi_len=len(src_data[j].split())
            en_len=len(trg_data[j].split())
            norm_src.astype(np.float32).tofile(file_dict["fi_vec_len{N}".format(N=fi_len)])
            print(src_data[j],file=file_dict["fi_sent_len{N}".format(N=fi_len)])
            norm_trg.astype(np.float32).tofile(file_dict["en_vec_len{N}".format(N=en_len)])
            print(trg_data[j],file=file_dict["en_sent_len{N}".format(N=en_len)])
            counter+=1
            if counter%100000==0:
                print("Vectorized {c} sentence pairs".format(c=counter))
#                print(type(norm_src[0].astype(np.float32)))
            
            
#            counter+=1
#            if counter==len(src_data):
#                break
#        if counter==len(src_data):
#            break
    for key,value in file_dict.items():
        value.close()

    
#    return src_vectors,trg_vectors


def rank_keras(src_vectors,trg_vectors,src_sentences,trg_sentences,verbose=True):
    
    src_data=[s.strip() for s in gzip.open(src_sentences,"rt")]
    trg_data=[s.strip() for s in gzip.open(trg_sentences,"rt")]
    
    src_vectors=np.fromfile(src_vectors,np.float32)
    src_vectors=src_vectors.reshape(int(len(src_vectors)/150),150)#[:1000,:]
    trg_vectors=np.fromfile(trg_vectors,np.float32)
    trg_vectors=trg_vectors.reshape(int(len(trg_vectors)/150),150).T
    
    print("#",src_vectors.shape,trg_vectors.shape,file=sys.stderr)
    
    results=[]
    
    # dot product
    slice_point=10000
    for i in range(0,src_vectors.shape[0],slice_point):
        print(i,file=sys.stderr)
        sim_matrix=np.dot(src_vectors[i:i+slice_point,:],trg_vectors)
        print(i,"dot product ready",file=sys.stderr)
    
        # argpartition
        partition_matrix=np.argpartition(sim_matrix,(-3000,-1))[:,-3000:]
        print(partition_matrix.shape,file=sys.stderr)
        print(i,"partition ready",file=sys.stderr)
    
        for j,row in enumerate(partition_matrix):
            if sim_matrix[j,row[-1]]<0.2:
                continue
            
#            results.append((i+j,[(sim_matrix[j,idx],idx) for idx in row]))
            print("source:",src_data[i+j])
            for idx in row:
                print(sim_matrix[j,idx],trg_data[idx])
            print()
#        print(i,"results ready for",len(results),"sentences",file=sys.stderr)
    
    
    return results
    
    
    
def rank_dictionary(keras_results,src_sentences,trg_sentences,verbose=True):

    f2e_dictionary=build_dictionary("lex.f2e", "uniq.train.tokens.fi.100K")
    e2f_dictionary=build_dictionary("lex.e2f", "uniq.train.tokens.en.100K")
    
    src_data=[s.strip() for s in gzip.open(src_sentences,"rt")]
    trg_data=[s.strip() for s in gzip.open(trg_sentences,"rt")]
    
    ranks=[]
    na=0
    all_scores=[]
    count=0
    print("Dictionary baseline",file=sys.stderr)
    
    for i, (src_sent_idx,pairs) in enumerate(keras_results):
        src_sent=src_data[src_sent_idx]
        english_transl=set()
        finnish_words=set(src_sent.lower().split())
        for w in finnish_words:
            if w in f2e_dictionary:
                english_transl.update(f2e_dictionary[w])

        combined=[]
        for j,(s,trg_sent_idx) in enumerate(pairs):  
            
            trg_sent=trg_data[trg_sent_idx]
            english_words=set(trg_sent.strip().lower().split())
            score=len(english_words&english_transl)/len(english_words) 
            finnish_transl=set()
            for w in english_words:
                if w in e2f_dictionary:
                    finnish_transl.update(e2f_dictionary[w])
            score2=len(finnish_words&finnish_transl)/len(finnish_words)
            avg=(s+score+score2)/3
            combined.append((avg,trg_sent))
        results=sorted(combined, key=lambda x:x[0], reverse=True)
        count+=1
        if count%10000==0:
            print(count,file=sys.stderr)
        if results[0][0]<0.2: # makes no sense to keep these...
            continue
        all_scores.append((results[0][0],src_sent,results))
        
        

        if verbose:
            print("Source:",i,src_sent)
            for s,txt in results[:10]:
                print(txt,s)
            print("*"*20)
            print()
            
    for (best_sim,src_sent,translations) in sorted(all_scores, key=lambda x:x[0], reverse=True):
        print("source:",src_sent)
        for (s,trg_sent) in translations[:20]:
            print(trg_sent,s)
        print("")
        

    print("# num:",len(all_scores),file=sys.stderr)


    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, default={n}'.format(n=1000))
    g.add_argument('--fi_len', type=int, help='Finnish matrix len')
    g.add_argument('--en_len', type=int, help='English matrix len')
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None:
        parser.print_help()
        sys.exit(1)


#    vectorize(args.vocabulary,args.model,"pbv4_ud.part-00.gz","encow14ax01.xml.gz",args.max_pairs)
#    vectorize(args.vocabulary,args.model,"data/all.test.fi.tokenized","data/all.test.en.tokenized")

#    for s in iter_wrapper("pbv4_ud.part-00.gz","encow14ax01.xml.gz",max_sent=1000):
#        pass

    keras_results=rank_keras("vdata_uniq/fi_vec_len{n}.npy".format(n=args.fi_len),"vdata_uniq/en_vec_len{n}.npy".format(n=args.en_len),"vdata_uniq/fi_sent_len{n}.txt.gz".format(n=args.fi_len),"vdata_uniq/en_sent_len{n}.txt.gz".format(n=args.en_len),verbose=False)
#    rank_dictionary(keras_results,"vdata_uniq/fi_sent_len{n}.txt.gz".format(n=args.fi_len),"vdata_uniq/en_sent_len{n}.txt.gz".format(n=args.en_len),verbose=False)
    
#    test("data/all.test.fi.tokenized","data/all.test.en.tokenized",args.model,args.vocabulary,args.max_pairs)
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


