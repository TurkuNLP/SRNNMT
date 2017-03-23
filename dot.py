import numpy as np
import sys
import math
import gzip
import time
import pickle
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import re
import array
from csr_csc_dot import csr_csc_dot_f
from sklearn.feature_extraction.text import CountVectorizer

from dictionary_baseline import build_dictionary



def build_translation_matrix(translation_dictionary,word2idx_orig,word2idx_foreign):
    # return sparse matrix which translates orig words into foreign
    column_size=len(word2idx_foreign) # foreign vocab size
    row=array.array('i')
    col=array.array('i')
    uniq=set()
    transl_row=array.array('i')
    transl_col=array.array('i')
    uniq_transl=set()
    for word,idx in sorted(word2idx_orig.items(), key=lambda x:x[1]):
        word=word.lower()
        if word not in translation_dictionary:
            continue
        for translation in translation_dictionary[word]:
            translation=translation.lower()
            if translation not in word2idx_foreign:
                continue
            row.append(idx)
            col.append(word2idx_foreign[translation])
    sparse=coo_matrix((np.ones(len(row),dtype=np.float32),(np.frombuffer(row,dtype=np.int32),np.frombuffer(col,dtype=np.int32))),shape=(len(word2idx_orig),column_size),dtype=np.float32)
    return sparse
    
    
def tokenize(text):    
    return text.split(" ")
    
def translate(data,dictionary):

    normalizer=np.zeros(len(data),dtype=np.float32)
    translations=[]
    for i,s in enumerate(data):
        s=s.split(" ")
        normalizer[i]=len(set(s))
        words=set()
        for w in s:
            if w not in dictionary:
                continue
            words|=dictionary[w]
        translations.append(" ".join(words))
    return translations,normalizer
    
def build_sparse_sklearn(data,translation_dictionary,word2idx,word2idx_translated):
    # sparse
    vectorizer=CountVectorizer(lowercase=True,binary=True,vocabulary=word2idx,analyzer="word",tokenizer=tokenize,dtype=np.float32)
    sparse=vectorizer.fit_transform(data)

    return sparse

    
def load_data(lang_fname):

    # load senteces, vectors and metadata
    sentences=[s.strip() for s in gzip.open(lang_fname+".gz","rt")]
    vectors=np.fromfile(lang_fname+".npy",np.float32)
    vectors=vectors.reshape(int(len(vectors)/150),150)
    metadata=np.fromfile(lang_fname+".meta",np.float32)
    metadata=metadata.reshape(int(len(metadata)/2),2)
    
    index_array=np.lexsort((metadata[:,1],metadata[:,0]),axis=0)
    sentences=[sentences[i] for i in index_array]
    vectors=vectors[index_array]
    metadata=metadata[index_array]
    
    return sentences,vectors,metadata
    
def process_metadata(m):

    d={}
    last=None # tuple=(cluster,length)
    for i,row in enumerate(m):
        cluster,length=row
        if (cluster,length) not in d: # this is new cluster, so mark start point and end point for last cluster
            if last!=None:
                d[last]=(d[last][0],i-1) # end point
                last=None
            d[(cluster,length)]=(i,None) # start point
            last=(cluster,length)
    else:
        if last:
            d[last]=(d[last][0],i) # end point for last item
            
    return d
    
def align_slices(src,trg,limit_length=(1,6),sentence_range=(5,30)):
    # limit_length:(min,max) --> limit trg sentence lengths compared to src length, first to consider is src+min, and last is src+max
    # sentence_range:(min,max) --> min and max sentence length we have in our data
    limited_src={} # e.g. no need to keep src_length=30 because we don't have trg_length=31
    aligned_trg={}
    for key,value in src.items():
        assert key not in aligned_trg
        if key[1]+limit_length[0]>sentence_range[1]:
            print("skipping src value:",key,file=sys.stderr)
            continue
        limited_src[key]=value
        aligned_trg[key]=(trg[(key[0],key[1]+limit_length[0])][0],trg[(key[0],min(key[1]+limit_length[1],sentence_range[1]))][1])
    return aligned_trg,limited_src

        
    
def get_slices(src_m,trg_m):
    # (cluster: from,to -- cluster,len: from,to)
    src=process_metadata(src_m)
    trg=process_metadata(trg_m)
    # now align these two, so that we can query trg slices using src keys, i.e. give me trg slice for src cluster 22 and sentence length 12 --> trg slice starts from (cluster22,length13) and ends to (cluster22,length18)
    aligned_trg,src=align_slices(src,trg)
    return src,aligned_trg

def main(src_fname,trg_fname,outfname,dictionary,dict_vocabulary):

    # file to write results
    out_combined=gzip.open(outfname+"-combined.txt.gz","wt",encoding="utf-8")
    out_keras=gzip.open(outfname+"-keras.txt.gz","wt",encoding="utf-8")
    out_baseline=gzip.open(outfname+"-baseline.txt.gz","wt",encoding="utf-8")
    
    
    ## read data (returns it already sorted)
    src_sentences,src_vectors,src_metadata=load_data(src_fname)
    trg_sentences,trg_vectors,trg_metadata=load_data(trg_fname)
    print("# Data sizes:",src_vectors.shape,trg_vectors.shape,file=sys.stderr)
    
    ## calculate slices (cluster: from,to -- cluster,len: from,to)
    src_slices,trg_slices=get_slices(src_metadata,trg_metadata) # trg_slices is aligned to src, it already knows clusters,sentence lengths and all


    ## translation dictionaries    
    f2e_dictionary=build_dictionary(dictionary+".f2e",dict_vocabulary+".fi")
    e2f_dictionary=build_dictionary(dictionary+".e2f",dict_vocabulary+".en")
    word2idx_fi={word.strip().lower(): i for i,word in enumerate(open(dict_vocabulary+".fi","rt",encoding="utf-8"))}
    word2idx_en={word.strip().lower(): i for i,word in enumerate(open(dict_vocabulary+".en","rt",encoding="utf-8"))}
    # translation matrices
    f2e_matrix=build_translation_matrix(f2e_dictionary,word2idx_fi,word2idx_en).tocsr()
    e2f_matrix=build_translation_matrix(e2f_dictionary,word2idx_en,word2idx_fi).tocsr()
    print("f2e translation matrix",f2e_matrix.shape,file=sys.stderr)
    print("e2f translation matrix",e2f_matrix.shape,file=sys.stderr)
    

    ## build sparse matrices
    print("# Building sparse matrices",file=sys.stderr)

    start=time.time()
    src_sparse=build_sparse_sklearn(src_sentences,f2e_dictionary,word2idx_fi,word2idx_en)
    src_normalizer=np.array([len(set(s.split(" "))) for s in src_sentences],dtype=np.float32)
    print(src_sparse.shape,src_normalizer.shape,file=sys.stderr)
    print("src sparse",time.time()-start,file=sys.stderr)
    start=time.time()
    trg_sparse=build_sparse_sklearn(trg_sentences,e2f_dictionary,word2idx_en,word2idx_fi)
    trg_normalizer=np.array([len(set(s.split(" "))) for s in trg_sentences],dtype=np.float32)
    print(trg_sparse.shape,trg_normalizer.shape,file=sys.stderr)
    print("trg sparse",time.time()-start,file=sys.stderr)
    
   
    ## dot product   
    print("# Running dot",file=sys.stderr)
        
    # iterate slices
    max_slice_point=1000
    for (src_key,(src_start,src_end)) in sorted(src_slices.items()):
        print("# source slice:",src_key,src_start,src_end,file=sys.stderr)
        trg_start,trg_end=trg_slices[src_key]
        print("# target slice:",trg_start,trg_end,file=sys.stderr)
        # take max_slice_point on src size, and whole slice on trg side
        
        #create output matrices
        sparse_dot_out=np.zeros((min(src_end+1-src_start,max_slice_point),trg_end+1-trg_start),dtype=np.float32)
        sparse_dot_out2=np.zeros((min(src_end+1-src_start,max_slice_point),trg_end+1-trg_start),dtype=np.float32)
        print("# sparse_dot_out",sparse_dot_out.shape,file=sys.stderr)
        
        #slice sparse trg now
        stime=time.time()
        trg_sparse_slice=trg_sparse[trg_start:trg_end+1,:] # slice original trg
        trg_translated_sparse_slice=(trg_sparse_slice*e2f_matrix).tocsc() # translate it
        trg_translated_sparse_slice.data=np.ones(len(trg_translated_sparse_slice.data),dtype=np.float32) # Force translated into binary
        trg_sparse_slice=trg_sparse_slice.tocsc() # convert original into csc
        trg_normalizer_slice=trg_normalizer.reshape((1,len(trg_normalizer)))[:,trg_start:trg_end+1] # slice normalizer
        print("Slicing trg sparse",time.time()-stime,file=sys.stderr)
    
        for i in range(src_start,src_end+1,max_slice_point): # iterate src slice 1000 at time
    
            ostart=time.time()
            print("slice:",src_key,"{}-{}".format(i,min(src_end+1,i+max_slice_point)),file=sys.stderr)
            ## dense dot
            start=time.time()
            sim_matrix=np.dot(src_vectors[i:min(src_end+1,i+max_slice_point),:],trg_vectors[trg_start:trg_end+1,:].T)
            print("dense dot product ready,",time.time()-start,file=sys.stderr)
            print(src_vectors[i:min(src_end+1,i+max_slice_point),:].shape,trg_vectors[trg_start:trg_end+1,:].T.shape,file=sys.stderr)
            
            ## sparse dot   
            # slice output matrix if i+max_slice_point>src_end+1 
            if i+max_slice_point>src_end+1:
                sparse_dot_out=sparse_dot_out[:sim_matrix.shape[0],:]
                sparse_dot_out2=sparse_dot_out2[:sim_matrix.shape[0],:]
        
            # original src, translated trg, normalized with src #unique_tokens
            start=time.time()       
            csr_csc_dot_f(i,min(src_end+1-i,max_slice_point),src_sparse,trg_translated_sparse_slice,sparse_dot_out)
            np.divide(sparse_dot_out,src_normalizer.reshape((len(src_normalizer),1))[i:min(src_end+1,i+max_slice_point),:],sparse_dot_out) # normalize
            # src translated, original trg, normalized with trg #unique_tokens
            tmp=src_sparse[i:min(src_end+1,i+max_slice_point),:]*f2e_matrix # translate original src slice        
            tmp.data=np.ones(len(tmp.data),dtype=np.float32) # force to binary
            csr_csc_dot_f(0,tmp.shape[0],tmp,trg_sparse_slice,sparse_dot_out2)
            np.divide(sparse_dot_out2,trg_normalizer_slice,sparse_dot_out2) # normalize
            print("full sparse dot ready",time.time()-start,file=sys.stderr)
            
            # sum sparse_dot_out and sparse_dot_out2, write results to sparse_dot_out
            np.add(sparse_dot_out,sparse_dot_out2,sparse_dot_out)
            # sum all three, write results to sparse_dot_out2
            np.add(sim_matrix,sparse_dot_out,sparse_dot_out2)
            
            # now sim_matrix has dense similarities, sparse_dot_out has baseline similarities, and sparse_dot_out2 has combined similarities
        
            argmaxs_keras=np.argmax(sim_matrix,axis=1)
            argmaxs_baseline=np.argmax(sparse_dot_out,axis=1)
            argmaxs_combined=np.argmax(sparse_dot_out2,axis=1)
            
            ## print results
            for j in range(argmaxs_keras.shape[0]): # all three should have the same shape
                # keras
                print(sim_matrix[j,argmaxs_keras[j]],src_sentences[i+j],trg_sentences[trg_start+argmaxs_keras[j]],sep="\t",file=out_keras,flush=True)
                # baseline
                print(sparse_dot_out[j,argmaxs_baseline[j]]/2.0,src_sentences[i+j],trg_sentences[trg_start+argmaxs_baseline[j]],sep="\t",file=out_baseline,flush=True)
                # baseline
                print(sparse_dot_out2[j,argmaxs_combined[j]]/3.0,src_sentences[i+j],trg_sentences[trg_start+argmaxs_combined[j]],sep="\t",file=out_combined,flush=True)
                
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
    g.add_argument('--fi_fname', type=str, help='Finnish matrix file name (without .gz or .npy extentions)')
    g.add_argument('--en_fname', type=str, help='English matrix file name (without .gz or .npy extentions)')
    g.add_argument('--dictionary', type=str, help='Dictionary file to build translation dictonaries, without .e2f or .f2e extensions')
    g.add_argument('--dict_vocabulary', type=str, help='Vocabulary files used to limit vocabulary when building translation dictonaries, without .fi or .en extensions')
    g.add_argument('--outfile', type=str, help='File to output results')
    
    args = parser.parse_args()

#    if args.model==None or args.vocabulary==None:
#        parser.print_help()
#        sys.exit(1)

    assert args.fi_fname.split(".",1)[-1]==args.en_fname.split(".",1)[-1]

    returns=main(args.fi_fname,args.en_fname,args.outfile,args.dictionary,args.dict_vocabulary)


    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


