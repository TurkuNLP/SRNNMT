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


def rank_keras(dirname,src_fname,trg_fname):
    
    src_data=[s.strip() for s in gzip.open(dirname+"/"+src_fname+".txt.gz","rt")]
    trg_data=[s.strip() for s in gzip.open(dirname+"/"+trg_fname+".txt.gz","rt")]
    
    src_vectors=np.fromfile(dirname+"/"+src_fname+".npy",np.float32)
    src_vectors=src_vectors.reshape(int(len(src_vectors)/150),150)#[:1000,:]
    trg_vectors=np.fromfile(dirname+"/"+trg_fname+".npy",np.float32)
    trg_vectors=trg_vectors.reshape(int(len(trg_vectors)/150),150).T
    
    print("#",src_vectors.shape,trg_vectors.shape,file=sys.stderr)
    
    out_sim=open(dirname+"/"+src_fname+"+"+trg_fname+".sim.npy","wb")
    out_idx=open(dirname+"/"+src_fname+"+"+trg_fname+".idx.npy","wb")
    
    results=[]
    
    # dot product
    slice_point=10000
    keep_point=3000
    for i in range(0,src_vectors.shape[0],slice_point):
        print(i,file=sys.stderr)
        sim_matrix=np.dot(src_vectors[i:i+slice_point,:],trg_vectors)
        print(i,"dot product ready",file=sys.stderr)
    
        # argpartition
        partition_matrix=np.argpartition(sim_matrix,(-keep_point,-1))[:,-keep_point:].astype(np.int32)
        #partition_matrix=partition_matrix
        print(partition_matrix.shape,file=sys.stderr)
        print(i,"partition ready",file=sys.stderr)
        
        # slice similarity matrix based on partition indices
        rows=np.array(range(sim_matrix.shape[0]))[:,np.newaxis]
        sliced_sim_matrix=sim_matrix[rows,partition_matrix]
        
        
        # ...print
        for x,(index_row,sim_row) in enumerate(zip(partition_matrix,sliced_sim_matrix)):
            index_row.tofile(out_idx)
            sim_row.tofile(out_sim)
            
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
    
    out_sim.close()
    out_idx.close()
    return results
    
    
    

    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
#    g.add_argument('-m', '--model', type=str, help='Give model name')
#    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, default={n}'.format(n=1000))
    g.add_argument('--fi_len', type=int, help='Finnish matrix len')
    g.add_argument('--en_len', type=int, help='English matrix len')
    
    args = parser.parse_args()

#    if args.model==None or args.vocabulary==None:
#        parser.print_help()
#        sys.exit(1)


#    keras_results=rank_keras("vdata_uniq/fi_vec_len{n}.npy".format(n=args.fi_len),"vdata_uniq/en_vec_len{n}.npy".format(n=args.en_len),"vdata_uniq/fi_sent_len{n}.txt.gz".format(n=args.fi_len),"vdata_uniq/en_sent_len{n}.txt.gz".format(n=args.en_len),verbose=False)

    keras_results=rank_keras("vdata_test","fi_len12","en_len12")

#    rank_dictionary(keras_results,"vdata_uniq/fi_sent_len{n}.txt.gz".format(n=args.fi_len),"vdata_uniq/en_sent_len{n}.txt.gz".format(n=args.en_len),verbose=False)
    
#    test("data/all.test.fi.tokenized","data/all.test.en.tokenized",args.model,args.vocabulary,args.max_pairs)
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


