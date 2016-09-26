from keras.models import Sequential, Graph, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
from keras.layers.recurrent import GRU
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import numpy as np
import sys
import math
import json

import data_dense

# load model
def load_model(mname):
    with open(mname+".json","r") as f:
        trained_model=model_from_json(f.read())
        trained_model.load_weights(mname+".h5")
        trained_model.layers.pop() # remove cosine and flatten layers
        trained_model.layers.pop()
        trained_model.outputs = [trained_model.get_layer('source_dense').output,trained_model.get_layer('target_dense').output] # define new outputs
        trained_model.layers[-1].outbound_nodes = [] # not sure if we need these...
        trained_model.layers[-2].outbound_nodes = []
        print(trained_model.summary())
        print(trained_model.outputs)
        
    return trained_model


def iter_wrapper(src_data,trg_data):
    for src_sent,trg_sent in zip(src_data,trg_data):
        yield (src_sent,trg_sent),1.0


def vectorize(voc_name,src_data,trg_data,mname):

    minibatch_size=100 
    ngrams=(4,) # TODO: read this from somewhere

    #Read vocabularies
    vs=data_dense.read_vocabularies(voc_name,"xxx","xxx",False,ngrams) 
    vs.trainable=False
    
    # load model
    trained_model=load_model(mname)
    output_size=trained_model.get_layer('source_dense').output_shape[1]
    max_sent_len=trained_model.get_layer('source_ngrams_{n}'.format(n=ngrams[0])).output_shape[1]
    
    # build matrices
    ms=data_dense.Matrices(minibatch_size,max_sent_len,ngrams)
    
    src_vectors=np.zeros((len(src_data),output_size))
    trg_vectors=np.zeros((len(src_data),output_size))

    # get vectors
    # for loop over minibatches
    counter=0    
    for i,(mx,targets) in enumerate(data_dense.fill_batch(minibatch_size,max_sent_len,vs,iter_wrapper(src_data,trg_data),ngrams)):
        src,trg=trained_model.predict(mx) # shape = (minibatch_size,gru_width)
        # loop over items in minibatch
        for j,(src_v,trg_v) in enumerate(zip(src,trg)):
            src_vectors[counter]=src_v/np.linalg.norm(src_v)
            trg_vectors[counter]=trg_v/np.linalg.norm(trg_v)
            counter+=1
            if counter==len(src_data):
                break
        if counter==len(src_data):
            break

    return src_vectors,trg_vectors
    
    
def rank(src_vectors,trg_vectors,src_data,trg_data,verbose=True):

    ranks=[]
    all_similarities=[] # list of sorted lists

    # run dot product
    for i in range(len(src_vectors)):
        sims=trg_vectors.dot(src_vectors[i])
        all_similarities.append(sims)  
        N=10
#        results=sorted(((sims[idx],idx,trg_data[idx]) for idx in np.argpartition(sims,-N-1)), reverse=True)#[-N-1:]), reverse=True)
        results=sorted(((sims[idx],idx,trg_data[idx]) for idx,s in enumerate(sims)), reverse=True)#[-N-1:]), reverse=True)
#        if results[0][0]<0.6:
#            continue
        result_idx=[idx for (sim,idx,txt) in results]
        ranks.append(result_idx.index(i)+1)
        if verbose:
            print("source:",i,src_data[i],np.dot(src_vectors[i],trg_vectors[i]))
            print("reference:",trg_data[i])
            print("rank:",result_idx.index(i)+1)
            for s,idx,txt in results[:10]:
                print(idx,s,txt)
            print("****")

    print("Keras:")
    print("Avg:",sum(ranks)/len(ranks))
    print("#num:",len(ranks))
    
    return all_similarities
    
    
def test(src_fname,trg_fname,mname,voc_name,max_pairs,verbose):

    # read sentences
    src_data=[]
    trg_data=[]
    for src_line,trg_line in data_dense.iter_data(src_fname,trg_fname,max_pairs=max_pairs):
        src_data.append(src_line.strip())
        trg_data.append(trg_line.strip())
        
    src_vectors,trg_vectors=vectorize(voc_name,src_data,trg_data,mname)
    similarities=rank(src_vectors,trg_vectors,src_data,trg_data,verbose=verbose)

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give model name')
    #g.add_argument('--cutoff', type=int, default=2, help='Frequency threshold, how many times an ngram must occur to be included? (default %(default)d)')
    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
    g.add_argument('--verbose', action='store_true', default=False, help='Give vocabulary file')
    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, default={n}'.format(n=1000))
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None:
        parser.print_help()
        sys.exit(1)

    test("data/all.test.new.fi.tokenized","data/all.test.new.en.tokenized",args.model,args.vocabulary,args.max_pairs,verbose=args.verbose)
#    test("data/wmttest.fi-en.fi.tokenized","data/wmttest.fi-en.en.tokenized",args.model,args.vocabulary,args.max_pairs)
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


