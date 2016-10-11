#from keras.models import Sequential, Graph, Model, model_from_json
#from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
#from keras.layers.recurrent import GRU
#from keras.callbacks import Callback,ModelCheckpoint
#from keras.layers.embeddings import Embedding
import numpy as np
import sys
import math
import json
import gzip
import conllutil3 as cu
import html
import time
import pickle
from scipy.sparse import csr_matrix,lil_matrix,csc_matrix,coo_matrix
import data_dense
import re
import hashlib

from test import load_model
from dictionary_baseline import build_dictionary


def build_sparse_matrices(data,translation_dictionary):
    # return sparse, and translated_sparse
    sparse_size=1000000
    normalizer=np.zeros(len(data),dtype=np.float32)
    row=[]
    col=[]
    values=[]
    transl_row=[]
    transl_col=[]
    transl_values=[]
    for i,sent in enumerate(data):
        words=set(sent.strip().lower().split())
        normalizer[i]=np.float32(len(words))
        for word in words:
            if word not in translation_dictionary:
                continue
            h=int(hashlib.sha224(word.encode("utf-8")).hexdigest(), 16)%sparse_size
            row.append(i)
            col.append(h)
            values.append(np.float32(1.0))
            for translation in translation_dictionary[word]:
                h=int(hashlib.sha224(translation.encode("utf-8")).hexdigest(), 16)%sparse_size
                transl_row.append(i)
                transl_col.append(h)
                transl_values.append(np.float32(1.0))
        if i!=0 and i%10000==0:
            print(i)
    sparse=coo_matrix((values,(row,col)),shape=(len(data),sparse_size),dtype=np.float32)
    translated_sparse=coo_matrix((transl_values,(transl_row,transl_col)),shape=(len(data),sparse_size),dtype=np.float32)
    return sparse.tocsr(), translated_sparse.tocsr(), normalizer


def rank_keras(dirname,src_fname,trg_fname,outfname):

    # file to write results
    outf=open(outfname,"wt",encoding="utf-8")

    
    # read data
    src_data=[s.strip() for s in gzip.open(dirname+"/"+src_fname+".txt.gz","rt")]#[:100000]
    src_vectors=np.fromfile("{D}/{F}.npy".format(D=dirname,F=src_fname),np.float32)
    src_vectors=src_vectors.reshape(int(len(src_vectors)/150),150)#[:100000,:]
    
    trg_data=[s.strip() for s in gzip.open(dirname+"/"+trg_fname+".txt.gz","rt")]#[:100000]
    trg_vectors=np.fromfile("{D}/{F}.npy".format(D=dirname,F=trg_fname),np.float32)
    trg_vectors=trg_vectors.reshape(int(len(trg_vectors)/150),150).T#[:100000,:].T
    
    print("#",src_vectors.shape,trg_vectors.shape,file=sys.stderr)
    
    # read translation dictionaries
    with open("f2e_dictionary.pickle", "rb") as f:
        f2e_dictionary=pickle.load(f)
    with open("e2f_dictionary.pickle", "rb") as f:
        e2f_dictionary=pickle.load(f)
    
    # build target sparse matrices    
    trg_sparse,trg_translated_sparse,trg_normalizer=build_sparse_matrices(trg_data,e2f_dictionary)
    
    trg_sparse=trg_sparse.T # transpose
    trg_translated_sparse=trg_translated_sparse.T
    
#    with open("trg_sparse.pickle", "wb") as f:
#        pickle.dump(trg_sparse,f)
#    with open("trg_translated_sparse.pickle", "wb") as f:
#        pickle.dump(trg_translated_sparse,f)
#    with open("trg_normalizer.pickle", "wb") as f:
#        pickle.dump(trg_normalizer,f)
#    with open("trg_sparse.pickle", "rb") as f:
#        trg_sparse=pickle.load(f)
#    with open("trg_translated_sparse.pickle", "rb") as f:
#        trg_translated_sparse=pickle.load(f)
#    with open("trg_normalizer.pickle", "rb") as f:
#        trg_normalizer=pickle.load(f)
    
#    out_sim=open("{D}_out/{SF}+{TF}.sim.npy".format(D=dirname,SF=src_fname,TF=trg_fname),"wb")
#    out_idx=open("{D}_out/{SF}+{TF}.idx.npy".format(D=dirname,SF=src_fname,TF=trg_fname),"wb")
    
    results=[]
    
    # dot product
    slice_point=500
    for i in range(0,src_vectors.shape[0],slice_point):
        ostart=time.time()
        print("slice:",i,file=sys.stderr)
        # keras dot
        start=time.time()
        sim_matrix=np.dot(src_vectors[i:i+slice_point,:],trg_vectors)
        end=time.time()
        print(i,"dot product ready,",end-start,file=sys.stderr)
        
        # dictionary dot
        
        # build source sparse matrices using slice of the data
        src_sparse,src_translated_sparse,src_normalizer=build_sparse_matrices(src_data[i:i+slice_point],f2e_dictionary)
        
        # fi orig, en transl, norm with fi_len
        start=time.time()
        en2fi=src_sparse.dot(trg_translated_sparse).toarray() # normalize with src_normalizer
        np.divide(en2fi,src_normalizer.reshape((len(src_normalizer),1)),en2fi)
        # fi transl, en orig, norm with en_len
        fi2en=src_translated_sparse.dot(trg_sparse).toarray()
        np.divide(fi2en,trg_normalizer.reshape((1,len(trg_normalizer))),fi2en)
        end=time.time()
        print("sparse dot ready",end-start)
        
        # sum all three, write results to sim_matrix
        start=time.time()
        np.add(sim_matrix,en2fi,sim_matrix)
        np.add(sim_matrix,fi2en,sim_matrix)
        end=time.time()
        print("average ready",end-start)
        
        start=time.time()
        argmaxs=np.argmax(sim_matrix,axis=1)
        end=time.time()
        print("argmax ready",end-start)
        
        for j in range(argmaxs.shape[0]):
            idx=argmaxs[j]
            sim=sim_matrix[j,idx]/3.0
            if sim<0.4:
                continue
            print(sim,src_data[i+j],trg_data[idx],sep="\t",file=outf)
            
        oend=time.time()
        print("Time:",oend-ostart)
        
        # argpartition
#        start=time.time()
#        partition_matrix=np.argpartition(sim_matrix,(-keep_point,-1))[:,-keep_point:].astype(np.int32)
        #partition_matrix=np.argsort(sim_matrix)[:,-keep_point:].astype(np.int32)
#        end=time.time()
        #partition_matrix=partition_matrix
#        print(partition_matrix.shape,file=sys.stderr)
#        print(i,"partition ready,",end-start,file=sys.stderr)
        
        # slice similarity matrix based on partition indices
#        start=time.time()
#        rows=np.array(range(sim_matrix.shape[0]))[:,np.newaxis]
#        sliced_sim_matrix=sim_matrix[rows,partition_matrix]
#        end=time.time()
#        print("sim sliced,",end-start)
        
#        start=time.time()
        # ...print
#        for x,(index_row,sim_row) in enumerate(zip(partition_matrix,sliced_sim_matrix)):
#            index_row.tofile(out_idx)
#            sim_row.tofile(out_sim)
#        end=time.time()
#        print("printing,",end-start)    
#            print("new source:",src_data[i+x],file=sys.stderr)
#            for idx,sim in zip(index_row,sim_row):
#                print(sim,trg_data[idx],file=sys.stderr)
#            print(file=sys.stderr)
            
            
            
            
#        for j,row in enumerate(partition_matrix):
#            if sim_matrix[j,row[-1]]<0.2:
#                continue
            
#            results.append((i+j,[(sim_matrix[j,idx],idx) for idx in row]))
#            print("source:",src_data[i+j])
#            for idx in row:
#                print(sim_matrix[j,idx],trg_data[idx])
#            print()
#        print(i,"results ready for",len(results),"sentences",file=sys.stderr)
    
#    out_sim.close()
#    out_idx.close()
    out.close()
    return results
    
    
    

    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
#    g.add_argument('-m', '--model', type=str, help='Give model name')
#    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
#    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, default={n}'.format(n=1000))
    g.add_argument('--fi_len', type=int, help='Finnish matrix len')
    g.add_argument('--en_len', type=int, help='English matrix len')
    g.add_argument('--outfile', type=str, help='File to output results')
    
    args = parser.parse_args()

#    if args.model==None or args.vocabulary==None:
#        parser.print_help()
#        sys.exit(1)



    keras_results=rank_keras("vdata_final","fi_len{N}".format(N=args.fi_len),"en_len{N}".format(N=args.en_len),args.outfile)


#    test("data/all.test.fi.tokenized","data/all.test.en.tokenized",args.model,args.vocabulary,args.max_pairs)
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


