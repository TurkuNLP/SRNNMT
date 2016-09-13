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

import data_dense

from test import load_model
from dictionary_baseline import build_dictionary


min_len=5
max_len=30

def read_fin_parsebank(fname,max_sent=10000):

#    sentences=[]
    counter=0
    for comm, sent in cu.read_conllu(gzip.open(fname,"rt",encoding="utf-8")):
        if min_len<=len(sent)<=max_len:
            txt=" ".join(line[cu.FORM] for line in sent)
            yield txt
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
    for sent in sent_reader(gzip.open(fname,"rt",encoding="utf-8")):
        if min_len<=len(sent)<=max_len:
            txt=" ".join(sent)
            yield txt
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
        file_dict["fi_sent_len{N}".format(N=i)]=gzip.open("vdata/fi_sent_len{N}.txt.gz".format(N=i),"wt",encoding="utf-8")
        file_dict["fi_vec_len{N}".format(N=i)]=open("vdata/fi_vec_len{N}.npy".format(N=i),"wb")
        file_dict["en_sent_len{N}".format(N=i)]=gzip.open("vdata/en_sent_len{N}.txt.gz".format(N=i),"wt",encoding="utf-8")
        file_dict["en_vec_len{N}".format(N=i)]=open("vdata/en_vec_len{N}.npy".format(N=i),"wb")
    

    minibatch_size=100 
    ngrams=(4,) # TODO: read this from somewhere

    #Read vocabularies
    vs=data_dense.read_vocabularies(voc_name,"xxx","xxx",False,ngrams) 
    vs.trainable=False
    
    # load model
    trained_model=load_model(mname)
    output_size=trained_model.get_layer('source_dense').output_shape[1]
    max_sent_len=trained_model.get_layer('source_ngrams_{n}'.format(n=ngrams[0])).output_shape[1]
    print(output_size,max_sent_len)
    
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

    ranks=[]
    all_similarities=[] # list of sorted lists
    
    src_data=[s.strip() for s in gzip.open(src_sentences,"rt")]
    trg_data=[s.strip() for s in gzip.open(trg_sentences,"rt")]
    
    src_vectors=np.fromfile(src_vectors,np.float32).reshape(len(src_data),150)
    trg_vectors=np.fromfile(trg_vectors,np.float32).reshape(len(trg_data),150)
    
    
    print("#",len(src_data),len(trg_data))
    
    to_keep=[]
    
    # dot product
    sim_matrix=np.dot(src_vectors,trg_vectors.T)
#    print("dot product ready")
    
    # argpartition
    partition_matrix=np.argpartition(sim_matrix,-3000)#[-N-1:]
#    print("partition ready")
    
    results=[]
    for i,row in enumerate(partition_matrix):
        results.append((src_data[i],[(sim_matrix[i,idx],trg_data[idx]) for idx in row[-3000:]]))
    
#    for i in range(5):
#        print(results[i][0],results[i][1][:5])
    
    return results
    
    
#    for i in range(len(src_vectors)):
#        sims=trg_vectors.dot(src_vectors[i])
#        all_similarities.append(sims)  
#        N=10
##        results=sorted(((sims[idx],idx,trg_data[idx]) for idx in np.argpartition(sims,-N-1)), reverse=True)#[-N-1:]), reverse=True)
#        results=sorted(((sims[idx],idx,trg_data[idx]) for idx,s in enumerate(sims)), reverse=True)#[-N-1:]), reverse=True)
#        if results[0][0]<0.6:
#            continue
#        result_idx=[idx for (sim,idx,txt) in results]
#        ranks.append(result_idx.index(i)+1)
#        to_keep.append((src_data[i],[(s,txt) for s,idx,txt in results[:1000]]))
#        if verbose:
#            print("source:",i,src_data[i],np.dot(src_vectors[i],trg_vectors[i]))
##            print("reference:",trg_data[i])
##            print("rank:",result_idx.index(i)+1)
#            for s,idx,txt in results[:10]:
#                print(idx,s,txt)
#            print("****")
#            print

#    print("Keras:")
#    print("Avg:",sum(ranks)/len(ranks))
#    print("#num:",len(ranks))
#    
##    return all_similarities
#    return to_keep
    
    
    
def rank_dictionary(keras_results,verbose=True):

    f2e_dictionary=build_dictionary("lex.f2e", "uniq.train.tokens.fi.100K")
    e2f_dictionary=build_dictionary("lex.e2f", "uniq.train.tokens.en.100K")
    
    ranks=[]
    na=0
    all_scores=[]   
    
    for i, (src_sent,pairs) in enumerate(keras_results):
        english_transl=set()
        finnish_words=set(src_sent.lower().split())
        for w in finnish_words:
            if w in f2e_dictionary:
                english_transl.update(f2e_dictionary[w])

        combined=[]
        for j,(s,trg_sent) in enumerate(pairs):  
            count=0
            english_words=set(trg_sent.strip().lower().split())
            score=len(english_words&english_transl)/len(english_words) 
#            scores.append((j,score/len(english_words)))
            finnish_transl=set()
            for w in english_words:
                if w in e2f_dictionary:
                    finnish_transl.update(e2f_dictionary[w])
            score2=len(finnish_words&finnish_transl)/len(finnish_words)
#            scores2.append((j,score2/len(finnish_words)))
            avg=(s+score+score2)/3
            combined.append((avg,trg_sent))
#        combined=[(x,(f+e)/2) for (x,f),(y,e) in zip(scores,scores2)]
        results=sorted(combined, key=lambda x:x[0], reverse=True)
#        if combined[0][0]<0.4:
#            continue
        all_scores.append((results[0][0],src_sent,results))
#        all_scores.append(combined)
#        if combined[i][1]==0.0: # TODO
#            ranks.append(len(src_data)/2)
#            na+=1
#            continue
#        result_idx=[idx for idx,score in results]
#        ranks.append(result_idx.index(i)+1)
        if verbose:
            print("Source:",i,src_sent)
#            print("Reference:",trg_data[i],combined[i][1])
#            print("Rank:",result_idx.index(i)+1)
            for s,txt in results[:10]:
                print(txt,s)
            print("*"*20)
            print()
            
    for (best_sim,src_sent,translations) in sorted(all_scores, key=lambda x:x[0], reverse=True):
        print("source:",src_sent)
        for (s,trg_sent) in translations[:10]:
            print(trg_sent,s)
        print("")
        
#    print("Dictionary baseline:")    
#    print("Avg:",sum(ranks)/len(ranks))
    print("# num:",len(all_scores))
#    print("n/a:",na)
    
#    return all_scores

    
    
def test(src_fname,trg_fname,mname,voc_name,max_pairs):

    # read sentences
    src_data=[]
    trg_data=[]
    for src_line,trg_line in data_dense.iter_data(src_fname,trg_fname,max_pairs=max_pairs):
        src_data.append(src_line.strip())
        trg_data.append(trg_line.strip())
        
    src_vectors,trg_vectors=vectorize(voc_name,src_data,trg_data,mname)
    similarities=rank(src_vectors,trg_vectors,src_data,trg_data)

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


    vectorize(args.vocabulary,args.model,"pbv4_ud.part-00.gz","encow14ax01.xml.gz",args.max_pairs)
#    vectorize(args.vocabulary,args.model,"data/all.test.fi.tokenized","data/all.test.en.tokenized")
    
#    to_keep=rank_keras("finnish_vectors.npy","english_vectors.npy","finnish_sentences.txt.gz","english_sentences.txt.gz",verbose=False)
#    results=rank_keras("vdata/fi_vec_len15.npy","vdata/en_vec_len15.npy","vdata/fi_sent_len15.txt.gz","vdata/en_sent_len15.txt.gz",verbose=False)


    keras_results=rank_keras("vdata/fi_vec_len{n}.npy".format(n=args.fi_len),"vdata/en_vec_len{n}.npy".format(n=args.en_len),"vdata/fi_sent_len{n}.txt.gz".format(n=args.fi_len),"vdata/en_sent_len{n}.txt.gz".format(n=args.en_len),verbose=False)
    rank_dictionary(keras_results,verbose=False)
    
#    test("data/all.test.fi.tokenized","data/all.test.en.tokenized",args.model,args.vocabulary,args.max_pairs)
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


