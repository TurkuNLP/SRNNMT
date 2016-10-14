import numpy as np
import sys
import math
import json
import gzip
import conllutil3 as cu
import html
import time
import pickle
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import data_dense
import re
import hashlib
import array
from csr_csc_dot import csr_csc_dot_f

from dictionary_baseline import build_dictionary


def build_sparse_matrices(data,translation_dictionary):
    # return sparse, and translated_sparse
    sparse_size=1000000
    normalizer=np.zeros(len(data),dtype=np.float32)
    row=array.array('i')
    col=array.array('i')
    transl_row=array.array('i')
    transl_col=array.array('i')
    for i,sent in enumerate(data):
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
        if i!=0 and i%10000==0:
            print(i)
    sparse=coo_matrix((np.ones(len(row),dtype=np.float32),(np.frombuffer(row,dtype=np.int32),np.frombuffer(col,dtype=np.int32))),shape=(len(data),sparse_size),dtype=np.float32)
    translated_sparse=coo_matrix((np.ones(len(transl_row),dtype=np.float32),(np.frombuffer(transl_row,dtype=np.int32),np.frombuffer(transl_col,dtype=np.int32))),shape=(len(data),sparse_size),dtype=np.float32)
    return sparse, translated_sparse, normalizer


def rank(dirname,src_fname,trg_fname,outfname):

    # file to write results
    out_combined=gzip.open(outfname+"-combined.txt.gz","wt",encoding="utf-8")
    out_keras=gzip.open(outfname+"-keras.txt.gz","wt",encoding="utf-8")
    out_baseline=gzip.open(outfname+"-baseline.txt.gz","wt",encoding="utf-8")
    

    
    # read data
    src_data=[s.strip() for s in gzip.open(dirname+"/"+src_fname+".txt.gz","rt")]#[:10100]
    src_vectors=np.fromfile("{D}/{F}.npy".format(D=dirname,F=src_fname),np.float32)
    src_vectors=src_vectors.reshape(int(len(src_vectors)/150),150)#[:10100,:]
    
    trg_data=[s.strip() for s in gzip.open(dirname+"/"+trg_fname+".txt.gz","rt")]#[:10100]
    trg_vectors=np.fromfile("{D}/{F}.npy".format(D=dirname,F=trg_fname),np.float32)
    trg_vectors=trg_vectors.reshape(int(len(trg_vectors)/150),150).T#[:10100,:].T
    
    print("#",src_vectors.shape,trg_vectors.shape,file=sys.stderr)
    
    # read translation dictionaries
    with open("f2e_dictionary.pickle", "rb") as f:
        f2e_dictionary=pickle.load(f)
    with open("e2f_dictionary.pickle", "rb") as f:
        e2f_dictionary=pickle.load(f)
        
#    f2e_dictionary=build_dictionary("lex.f2e","uniq.train.tokens.fi.100K")
#    e2f_dictionary=build_dictionary("lex.e2f","uniq.train.tokens.en.100K")
    
    # build target sparse matrices    
    trg_sparse,trg_translated_sparse,trg_normalizer=build_sparse_matrices(trg_data,e2f_dictionary)
    trg_sparse=trg_sparse.tocsc() # csc
    trg_translated_sparse=trg_translated_sparse.tocsc()
    
    # build source sparse matrices
    src_sparse,src_translated_sparse,src_normalizer=build_sparse_matrices(src_data,f2e_dictionary)
    src_sparse=src_sparse.tocsr() # csr
    src_translated_sparse=src_translated_sparse.tocsr()
    
#    with open("trg_sparse.pickle", "wb") as f:
#        pickle.dump(trg_sparse,f)
#    with open("trg_translated_sparse.pickle", "wb") as f:
#        pickle.dump(trg_translated_sparse,f)
#    with open("trg_normalizer.pickle", "wb") as f:
#        pickle.dump(trg_normalizer,f)
        
#    with open("src_sparse.pickle", "wb") as f:
#        pickle.dump(src_sparse,f)
#    with open("src_translated_sparse.pickle", "wb") as f:
#        pickle.dump(src_translated_sparse,f)
#    with open("src_normalizer.pickle", "wb") as f:
#        pickle.dump(src_normalizer,f)
        
        
#    with open("trg_sparse.pickle", "rb") as f:
#        trg_sparse=pickle.load(f)
#    with open("trg_translated_sparse.pickle", "rb") as f:
#        trg_translated_sparse=pickle.load(f)
#    with open("trg_normalizer.pickle", "rb") as f:
#        trg_normalizer=pickle.load(f)
        
#    with open("src_sparse.pickle", "rb") as f:
#        src_sparse=pickle.load(f)
#    with open("src_translated_sparse.pickle", "rb") as f:
#        src_translated_sparse=pickle.load(f)
#    with open("src_normalizer.pickle", "rb") as f:
#        src_normalizer=pickle.load(f)
    
    slice_point=500
    
    sparse_dot_out=np.zeros((slice_point,len(trg_data)),dtype=np.float32)
    sparse_dot_out2=np.zeros((slice_point,len(trg_data)),dtype=np.float32)
    print("# sparse_dot_out",sparse_dot_out.shape)
    
    # dot product
    for i in range(0,src_vectors.shape[0],slice_point):
    
        ostart=time.time()
        print("slice:",i,file=sys.stderr)
        # keras dot
        start=time.time()
        sim_matrix=np.dot(src_vectors[i:i+slice_point,:],trg_vectors)
        end=time.time()
        print(i,"dot product ready,",end-start,file=sys.stderr)
        
        
        
        # sparse dot
        
        # slice out matrix if i+slice_point > len(src_data)
        if i+slice_point>len(src_data):
            sparse_dot_out=sparse_dot_out[:sim_matrix.shape[0],:]
            sparse_dot_out2=sparse_dot_out2[:sim_matrix.shape[0],:]
        
        # fi orig, en transl, norm with fi_len
        start=time.time()
        csr_csc_dot_f(i,slice_point,src_sparse,trg_translated_sparse,sparse_dot_out)
        print("# sparse_dot_out",sparse_dot_out.shape,file=sys.stderr)
        print("first sparse dot ready",time.time()-end,file=sys.stderr)
        np.divide(sparse_dot_out,src_normalizer.reshape((len(src_normalizer),1))[i:i+slice_point,:],sparse_dot_out)
        print("first normalize ready",time.time()-end,file=sys.stderr)
        # fi transl, en orig, norm with en_len
        csr_csc_dot_f(i,slice_point,src_translated_sparse,trg_sparse,sparse_dot_out2)
        print("second sparse dot ready",time.time()-end,file=sys.stderr)
        np.divide(sparse_dot_out2,trg_normalizer.reshape((1,len(trg_normalizer))),sparse_dot_out2)
        end=time.time()
        print("full sparse dot ready",end-start,file=sys.stderr)
        
        # sum sparse_dot_out and sparse_dot_out2, write results to sparse_dot_out
        start=time.time()
        np.add(sparse_dot_out,sparse_dot_out2,sparse_dot_out)
        # sum all three, write results to sparse_dot_out2
        np.add(sim_matrix,sparse_dot_out,sparse_dot_out2)
        end=time.time()
        print("average ready",end-start,file=sys.stderr)
        
        # now sim_matrix has keras similarities, sparse_dot_out has baseline similarities, and sparse_dot_out2 has combined similarities
        
        start=time.time()
        argmaxs_keras=np.argmax(sim_matrix,axis=1)
        argmaxs_baseline=np.argmax(sparse_dot_out,axis=1)
        argmaxs_combined=np.argmax(sparse_dot_out2,axis=1)
        end=time.time()
        print("argmax ready",end-start,file=sys.stderr)
        
        # print
        for j in range(argmaxs_keras.shape[0]): # all three should have the same shape
            # keras
            print(sim_matrix[j,argmaxs_keras[j]],src_data[i+j],trg_data[argmaxs_keras[j]],sep="\t",file=out_keras,flush=True)
            # baseline
            print(sparse_dot_out[j,argmaxs_baseline[j]]/2.0,src_data[i+j],trg_data[argmaxs_baseline[j]],sep="\t",file=out_baseline,flush=True)
            # baseline
            print(sparse_dot_out2[j,argmaxs_combined[j]]/3.0,src_data[i+j],trg_data[argmaxs_combined[j]],sep="\t",file=out_combined,flush=True)
            
        oend=time.time()
        print("Time:",oend-ostart,file=sys.stderr)
        print("",file=sys.stderr,flush=True)
            

    out_combined.close()
    out_keras.close()
    out_baseline.close()
    return 0
    
    
    

    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('--fi_fname', type=str, help='Finnish matrix file name (without .txt or .npy extentions)')
    g.add_argument('--en_fname', type=str, help='English matrix file name (without .txt or .npy extentions)')
    g.add_argument('--outfile', type=str, help='File to output results')
    
    args = parser.parse_args()

#    if args.model==None or args.vocabulary==None:
#        parser.print_help()
#        sys.exit(1)



    returns=rank("vdata_final",args.fi_fname,args.en_fname,args.outfile)


    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


